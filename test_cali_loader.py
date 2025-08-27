from cali_data_loader_eos import get_cali_stream

stream = get_cali_stream("v8")

for i, batch in enumerate(stream):    
    print(batch["eos_pos"].shape)
    if i > 3:
        break
