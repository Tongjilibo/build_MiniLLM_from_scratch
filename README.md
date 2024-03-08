# build_llm_from_scratch

- 监控训练期间的显存，内存和cpu
```python
from bert4torch.snippets import watch_system_state
watch_system_state(log_dir='./ckpt/states', pid=3490691)
```