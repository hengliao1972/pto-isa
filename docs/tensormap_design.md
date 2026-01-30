# TensorMap 设计与实现详解

## 1. 概述

TensorMap 是 PTO Runtime2 中用于追踪 **生产者-消费者关系** 的核心数据结构。它的主要功能是：

1. **依赖发现**：当提交新任务时，查找输入 tensor 的生产者任务
2. **重叠检测**：支持对 view、reshape、transpose 等操作后的 tensor 进行正确的依赖匹配
3. **高效管理**：O(1) 插入，惰性失效，链截断优化

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorMap 在运行时中的角色                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Orchestrator                     TensorMap                     │
│   ┌───────────┐                   ┌──────────────┐              │
│   │ submit    │ ──── lookup ────> │ region →     │              │
│   │ task(B)   │ <── producer_id ──│ producer_id  │              │
│   └───────────┘                   └──────────────┘              │
│        │                                 ▲                       │
│        │                                 │                       │
│        └──────── insert ─────────────────┘                       │
│                (B的output)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据结构

### 2.1 基础 TensorMap (PTO2TensorMap)

用于简单的 1D 区域匹配（基于 base_ptr + offset + size）。

```c
typedef struct {
    // 哈希表 buckets
    int32_t* buckets;             // 指向 entry_pool 的偏移量 (-1 = 空)
    int32_t  num_buckets;         // 必须是 2 的幂（快速取模）
    
    // Ring Buffer 条目池
    PTO2TensorMapEntry* entry_pool;
    int32_t pool_size;            // 池容量
    int32_t pool_head;            // 下一个分配位置
    
    // 每任务条目链表（用于清理）
    int32_t* task_entry_head;     // 每任务的条目头
    
    // 失效阈值
    int32_t last_task_alive;      // 低于此 ID 的任务已退休
} PTO2TensorMap;

// 条目结构
typedef struct {
    PTO2TensorRegion region;      // base_ptr, tile_index, offset, size
    int32_t producer_task_id;     // 生产者任务 ID
    int32_t next_in_bucket;       // 桶内链表
    int32_t next_in_task;         // 任务内链表
    bool in_bucket;               // 是否在桶中
} PTO2TensorMapEntry;
```

### 2.2 扩展 TensorMap (PTO2TensorMapEx)

支持多维 tensor 的 view/reshape/transpose 操作，使用 **边界盒（Bounding Box）** 进行重叠检测。

```c
typedef struct {
    // 原始存储信息
    void* raw_base;               // 原始 tensor 基地址
    int64_t raw_total_size;       // 原始 tensor 总大小
    
    // 边界盒（快速重叠检测）
    int64_t min_byte_offset;      // 最小字节偏移
    int64_t max_byte_offset;      // 最大字节偏移
    
    // 完整布局信息（精确检测可选）
    int64_t storage_offset;       // 存储偏移
    int64_t shape[PTO2_MAX_TENSOR_DIM];
    int64_t strides[PTO2_MAX_TENSOR_DIM];
    int32_t ndim;
    
    int32_t producer_task_id;
    bool is_deep_copy;            // 是否深拷贝（独立存储）
    // ... 链表指针
} PTO2TensorMapEntryEx;
```

---

## 3. 哈希策略：仅按 base_ptr 哈希

### 3.1 关键设计决策

```c
uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, PTO2TensorRegion* region) {
    // ========== 关键：仅按 base_ptr 哈希 ==========
    // 
    // 为了正确检测重叠，同一 base tensor 的所有区域
    // 必须在同一个哈希桶中！
    //
    uint64_t key = (uint64_t)(uintptr_t)region->base_ptr;
    
    // 位混合提高分布
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    return (uint32_t)(key & (tm->num_buckets - 1));
}
```

### 3.2 为什么不能包含 offset？

```
如果哈希包含 offset：
  Region A: base=X, offset=0   → bucket 5
  Region B: base=X, offset=128 → bucket 12  ❌ 无法检测重叠！

仅按 base_ptr 哈希：
  Region A: base=X, offset=0   → bucket 5
  Region B: base=X, offset=128 → bucket 5   ✓ 同一桶，可以比较！
```

---

## 4. 重叠检测算法

### 4.1 基础版本：1D 区间重叠

```c
bool pto2_region_overlap(PTO2TensorRegion* a, PTO2TensorRegion* b) {
    // 1. 必须是同一 base tensor
    if (a->base_ptr != b->base_ptr) return false;
    
    // 2. 必须是同一 tile（不同 tile 不重叠）
    if (a->tile_index != b->tile_index) return false;
    
    // 3. 区间重叠检测：[start_a, end_a) ∩ [start_b, end_b) ≠ ∅
    int32_t a_start = a->offset;
    int32_t a_end = a_start + a->size;
    int32_t b_start = b->offset;
    int32_t b_end = b_start + b->size;
    
    // 重叠条件：(a_start < b_end) AND (b_start < a_end)
    return (a_start < b_end) && (b_start < a_end);
}
```

### 4.2 扩展版本：边界盒重叠（支持 view/reshape/transpose）

```c
bool pto2_tensormapex_overlap(const PTO2LogicalTensor* tensor, 
                               const PTO2TensorMapEntryEx* entry) {
    // 1. 不同 raw storage 不重叠
    if (tensor->raw_base != entry->raw_base) return false;
    
    // 2. 边界盒交集检测
    // 重叠条件：(a.min <= b.max) AND (b.min <= a.max)
    return (tensor->min_byte_offset <= entry->max_byte_offset) &&
           (entry->min_byte_offset <= tensor->max_byte_offset);
}
```

### 4.3 边界盒计算

对于多维 tensor，边界盒是包含所有元素的最小连续内存范围：

```c
void pto2_logical_tensor_get_bounding_box(
    const PTO2LogicalTensor* tensor,
    int64_t* out_min,
    int64_t* out_max
) {
    // min_offset = storage_offset + Σ min(0, (shape[d]-1)*strides[d])
    // max_offset = storage_offset + Σ max(0, (shape[d]-1)*strides[d])
    
    int64_t min_offset = tensor->storage_offset;
    int64_t max_offset = tensor->storage_offset;
    
    for (int32_t d = 0; d < tensor->ndim; d++) {
        int64_t extent = (tensor->shape[d] - 1) * tensor->strides[d];
        if (extent >= 0) {
            max_offset += extent;
        } else {
            min_offset += extent;  // 负 stride
        }
    }
    
    *out_min = min_offset;
    *out_max = max_offset + tensor->elem_size - 1;  // 包含最后一个元素
}
```

---

## 5. 对 View/Reshape/Transpose 的支持

### 5.1 Tensor 提取类型

```c
typedef enum {
    PTO2_TENSOR_RAW,              // 原始 tensor（拥有存储）
    PTO2_TENSOR_SHALLOW_VIEW,     // 浅提取：view/slice
    PTO2_TENSOR_SHALLOW_RESHAPE,  // 浅提取：reshape
    PTO2_TENSOR_SHALLOW_TRANSPOSE,// 浅提取：transpose
    PTO2_TENSOR_DEEP_VIEW,        // 深提取：clone
    PTO2_TENSOR_DEEP_CONTIGUOUS,  // 深提取：contiguous
} PTO2TensorExtractionType;
```

### 5.2 浅提取（共享存储）→ 需要重叠检测

```
原始 Tensor A:  [0, 1, 2, 3, 4, 5, 6, 7]  raw_base = 0x1000
                ├────────────────────────┤

View B = A[2:6]: [2, 3, 4, 5]             raw_base = 0x1000
                    ├────────┤             storage_offset = 2 * elem_size

View C = A[4:8]: [4, 5, 6, 7]             raw_base = 0x1000
                        ├────────┤         storage_offset = 4 * elem_size

B 和 C 重叠！因为：
- 同一 raw_base (0x1000)
- 边界盒交集：[2*elem, 5*elem] ∩ [4*elem, 7*elem] = [4*elem, 5*elem] ≠ ∅
```

### 5.3 深提取（独立存储）→ 无需重叠检测

```c
// Clone 创建新的独立存储
bool pto2_logical_tensor_clone(src, dst, new_base) {
    // dst->raw_base = new_base (不同于 src->raw_base)
    // 因此不会与 src 重叠
}
```

### 5.4 Transpose 支持

Transpose 只改变 strides，不改变 raw_base：

```c
bool pto2_logical_tensor_transpose(src, dst, perm) {
    // 继承 raw_base 和 storage_offset
    dst->raw_base = src->raw_base;
    dst->storage_offset = src->storage_offset;
    
    // 重排 shape 和 strides
    for (int d = 0; d < src->ndim; d++) {
        dst->shape[d] = src->shape[perm[d]];
        dst->strides[d] = src->strides[perm[d]];
    }
    
    // 重新计算边界盒
    pto2_logical_tensor_update_bounding_box(dst);
}
```

---

## 6. 惰性失效与链截断优化

### 6.1 条目有效性检查

```c
static inline bool pto2_tensormap_entry_valid(PTO2TensorMap* tm, 
                                               PTO2TensorMapEntry* entry) {
    // 任务 ID >= last_task_alive 则有效
    return entry->producer_task_id >= tm->last_task_alive;
}
```

### 6.2 链截断优化

由于新条目总是插入链头（task_id 降序），一旦遇到失效条目，后续全部失效：

```c
int32_t pto2_tensormap_lookup(PTO2TensorMap* tm, PTO2TensorRegion* region) {
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    
    while (offset >= 0) {
        PTO2TensorMapEntry* entry = &tm->entry_pool[offset];
        
        // 检查有效性
        if (!pto2_tensormap_entry_valid(tm, entry)) {
            // ========== 链截断 ==========
            // 后续条目全部失效，直接截断
            *prev_ptr = -1;
            
            // 标记截断条目
            while (offset >= 0) {
                PTO2TensorMapEntry* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            
            return -1;  // 未找到
        }
        
        // 检查重叠
        if (pto2_region_overlap(&entry->region, region)) {
            return entry->producer_task_id;  // 找到！
        }
        
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return -1;  // 未找到
}
```

---

## 7. 查找所有重叠生产者

某些情况下，一个消费者可能依赖多个生产者（多个任务写入重叠区域）：

```c
int32_t pto2_tensormapex_lookup_all(PTO2TensorMapEx* tm, 
                                     const PTO2LogicalTensor* tensor,
                                     int32_t* producer_ids,
                                     int32_t max_producers) {
    uint32_t bucket = pto2_tensormapex_hash(tm, tensor);
    int32_t offset = tm->buckets[bucket];
    int32_t count = 0;
    
    while (offset >= 0 && count < max_producers) {
        PTO2TensorMapEntryEx* entry = &tm->entry_pool[offset];
        
        if (!pto2_tensormapex_entry_valid(tm, entry)) {
            // 链截断
            break;
        }
        
        if (pto2_tensormapex_overlap(tensor, entry)) {
            // 去重检查（同一生产者可能有多个输出）
            bool duplicate = false;
            for (int32_t i = 0; i < count; i++) {
                if (producer_ids[i] == entry->producer_task_id) {
                    duplicate = true;
                    break;
                }
            }
            
            if (!duplicate) {
                producer_ids[count++] = entry->producer_task_id;
            }
        }
        
        offset = entry->next_in_bucket;
    }
    
    return count;  // 返回找到的生产者数量
}
```

---

## 8. Ring Buffer 池管理

### 8.1 分配新条目

```c
void pto2_tensormap_insert(PTO2TensorMap* tm, PTO2TensorRegion* region, 
                            int32_t producer_task_id) {
    // 从 ring buffer 池分配
    int32_t entry_offset = tm->pool_head;
    PTO2TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    
    // 前进池头（环绕）
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;
    
    // ========== 关键：移除旧条目的桶链接 ==========
    // 即使条目已失效，它仍在桶链中，必须先移除
    if (entry->in_bucket) {
        pto2_tensormap_remove_from_bucket(tm, entry);
    }
    
    // 初始化新条目
    entry->region = *region;
    entry->producer_task_id = producer_task_id;
    
    // 插入桶头
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;
    
    // 链接到任务条目链表
    int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
    entry->next_in_task = tm->task_entry_head[task_slot];
    tm->task_entry_head[task_slot] = entry_offset;
}
```

---

## 9. 复杂度分析

| 操作 | 平均复杂度 | 最坏复杂度 |
|------|-----------|-----------|
| 插入 | O(1) | O(1) |
| 查找 | O(有效条目数 / 桶数) | O(有效条目数) |
| 清理 | O(退休任务的条目数) | O(条目总数) |

- **插入**：总是 O(1)，插入桶头
- **查找**：平均 O(chain_length)，链截断优化加速
- **清理**：惰性失效 + 定期显式清理

---

## 10. 与设计文档的一致性

当前实现完全符合 `runtime_buffer_manager_methods.md` 的设计：

| 设计要求 | 实现状态 |
|---------|---------|
| Ring buffer 池管理 | ✓ 使用 pool_head 环绕分配 |
| 惰性失效 | ✓ 通过 last_task_alive 阈值 |
| 链截断优化 | ✓ 遇到失效条目截断整个尾部 |
| 仅按 base_ptr 哈希 | ✓ 确保同一 tensor 的所有区域在同一桶 |
| 重叠检测 | ✓ 1D 区间和边界盒两种模式 |
| view/reshape/transpose 支持 | ✓ 通过 LogicalTensor 和边界盒 |
| 多生产者查找 | ✓ lookup_all 返回所有重叠生产者 |

---

## 11. 使用示例

```c
// 初始化
PTO2TensorMapEx tm;
pto2_tensormapex_init_default(&tm);

// 任务 A 生产 tensor
PTO2LogicalTensor output_A;
pto2_logical_tensor_init_raw(&output_A, buffer, shape, 2, sizeof(float));
pto2_tensormapex_insert(&tm, &output_A, task_id_A);

// 任务 B 的输入是 A 的 view
PTO2LogicalTensor input_B;
pto2_logical_tensor_view(&output_A, &input_B, start, new_shape, 2);

// 查找生产者
int32_t producer = pto2_tensormapex_lookup(&tm, &input_B);
// producer == task_id_A (因为 view 与原 tensor 重叠)

// 清理
pto2_tensormapex_destroy(&tm);
```

---

## 12. 混合重叠检测（Hybrid Overlap Detection）

### 12.1 设计动机

传统的重叠检测方法各有优缺点：

| 方法 | 优点 | 缺点 |
|------|------|------|
| Bounding Box | O(1) 快速 | 对非连续 tensor 有误报 |
| GCD 方法 | 100% 精确 | O(ndim) 较慢 |

**混合方法** 结合两者优势：
- 对 **简单 tensor** (连续) 使用 Bounding Box（快速且精确）
- 对 **复杂 tensor** (非连续) 使用 GCD（精确无误报）

### 12.2 Tensor 复杂度分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tensor 复杂度分类                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Tensor                                                         │
│     │                                                            │
│     ├── is_contiguous = true ──> Simple Tensor                  │
│     │     • 内存连续，无 gap                                     │
│     │     • Bounding Box = 实际访问范围                          │
│     │     • 检测结果：精确，无误报                                │
│     │                                                            │
│     └── is_contiguous = false ──> Complex Tensor                │
│           • 内存有 gap（如 transpose、strided view）             │
│           • Bounding Box 包含未访问区域                          │
│           • 检测结果：可能误报，需 GCD 验证                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 混合检测算法

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合检测流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   检测 overlap(A, B)                                            │
│     │                                                            │
│     ├─── A.raw_base ≠ B.raw_base ──> 不重叠 (不同存储)          │
│     │                                                            │
│     ├─── Bounding Box 不相交 ──> 不重叠 (快速排除)               │
│     │                                                            │
│     ├─── A.is_simple && B.is_simple ──> 重叠确认                │
│     │     (两者都是连续的，Bounding Box 结果精确)                │
│     │                                                            │
│     └─── 至少一个 Complex ──> GCD 精确检测                       │
│           (非连续 tensor 需要验证实际是否重叠)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.4 复杂度分析

| Tensor A | Tensor B | 检测方法 | 时间复杂度 |
|----------|----------|----------|------------|
| Simple | Simple | Bounding Box | O(1) |
| Simple | Complex | GCD | O(ndim) |
| Complex | Simple | GCD | O(ndim) |
| Complex | Complex | GCD | O(ndim²) |

**性能优势**：
- 大多数实际场景中 tensor 是连续的，走 O(1) 快速路径
- 只有涉及 transpose、非连续 view 时才需要 GCD
- Bounding Box 始终作为第一道过滤器

### 12.5 API

```c
/**
 * 混合重叠检测（推荐使用）
 * - 自动选择最优检测方法
 * - 返回结果 100% 精确，无误报
 */
bool pto2_logical_tensor_overlap_hybrid(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Tensor 与 TensorMapEntry 混合检测
 */
bool pto2_tensor_entry_overlap_hybrid(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);
```

### 12.6 数据结构支持

**TensorMapEntryEx** 新增 `is_simple` 字段：

```c
typedef struct PTO2TensorMapEntryEx {
    // ... 其他字段 ...
    
    bool is_deep_copy;   // 是否深拷贝
    bool is_simple;      // 是否简单 tensor（连续）
} PTO2TensorMapEntryEx;
```

**Insert 时记录**：

```c
void pto2_tensormapex_insert(...) {
    // ...
    entry->is_simple = tensor->is_contiguous;
    // ...
}
```

### 12.7 GCD 方法详解

对于非连续 tensor，使用 GCD（最大公约数）算法精确检测重叠：

**数学原理**：
```
Tensor A 访问的字节偏移: offset_A + i * stride_A  (0 <= i < size_A)
Tensor B 访问的字节偏移: offset_B + j * stride_B  (0 <= j < size_B)

重叠存在当且仅当存在整数 i, j 满足:
  offset_A + i * stride_A = offset_B + j * stride_B

变形: i * stride_A - j * stride_B = offset_B - offset_A = delta

根据数论，整数解存在当且仅当: gcd(stride_A, stride_B) | delta
```

**示例：消除误报**

```
Tensor A: offset=0, stride=8, size=4  -> 访问 [0, 8, 16, 24]
Tensor B: offset=4, stride=8, size=4  -> 访问 [4, 12, 20, 28]

Bounding Box:
  A: [0, 24]
  B: [4, 28]
  交集: [4, 24] 非空 -> 误报"重叠"

GCD 检测:
  delta = 4 - 0 = 4
  gcd(8, 8) = 8
  4 % 8 ≠ 0 -> 无整数解 -> 不重叠 ✓
```

### 12.8 实现状态

| 功能 | 状态 |
|------|------|
| is_simple 字段 | ✓ 已添加到 TensorMapEntryEx |
| hybrid 检测函数 | ✓ 已实现 |
| TensorMap 集成 | ✓ lookup 使用 hybrid |
| 1D GCD 检测 | ✓ 完整实现 |
| 多维 GCD 检测 | ✓ 降维到 1D 处理 |
