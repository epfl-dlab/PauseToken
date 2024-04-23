# Run the sft script 

```
python src/DPO/sft.py --model-name=models/Llama-2-7b-hf --n-epochs=3 --batch-size=8 --logging-steps=50 --use-peft=true --max-length=128 --save-steps=500 --eval-steps=3000 
```
