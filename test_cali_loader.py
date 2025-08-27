from cali_data_loader_eos import get_cali_stream

stream = get_cali_stream("v8", batch_size=4)

print(stream)
for i, batch in enumerate(stream):
    print(batch)
