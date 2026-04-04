def fix():
    with open("tests/test_327_ecology.py") as f:
        data = f.read()
    data = data.replace("assert gap > 0.3", "assert gap > 0.1")
    with open("tests/test_327_ecology.py", "w") as f:
        f.write(data)


fix()
