# NLP_Summarization

epoch  - 3, samples/batchsize - 32000/32

| -                                | -             | -             | -              | -              | -              | -              |
| -------------------------------- | ------------- | ------------- | -------------- | -------------- | -------------- | -------------- |
| WORKERS                          | 8             | 4             | 4              | 4              | 4              | 4              |
| NON_BLOCKING                     | False         | False         | False          | False          | True           | True           |
| DTYPE                            | Float32Tensor | Float32Tensor | BFloat16Tensor | BFloat16Tensor | BFloat16Tensor | BFloat16Tensor |
| WARMUP (decreased memory by 1gb) | False         | False         | False          | True           | True           | True           |
| TF32                             | False         | False         | False          | False          | False          | True           |
| TIME(s)(sum)                     | 70            | 69            | 55             | 54             | 52             | 52             |
| IT/S(mean)                       | 42.2          | 42.7          | 53.9           | 55.1           | 56             | 56.4           |
| LOSS(mean)                       | 0.006         | 0.006         | 0.006          | 0.006          | 0.006          | 0.006          |
| VRAM                             | +3.5gb        | +3.5gb        | +1.9           | +1.1           | +1.1           | +1.1           |
