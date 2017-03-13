k = 60000000
with open("test.txt", 'r') as f:
    with open("test_with_id.txt", 'w') as out:
        for line in f:
            out.write(str(k) + "\t" +  line)
            k += 1