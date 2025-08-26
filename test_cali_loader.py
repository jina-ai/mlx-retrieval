from cali_data_loader_eos import get_cali_stream

stream = get_cali_stream("v7", batch_size=4)

print(stream)
for i, batch in enumerate(stream):
    print(batch)
    if i > 2:
        break
