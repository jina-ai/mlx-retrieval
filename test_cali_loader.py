from cali_data_loader import get_cali_stream

stream = get_cali_stream("v7", batch_size=4)
for i, batch in enumerate(stream):
    print(i)
