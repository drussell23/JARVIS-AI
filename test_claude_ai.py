def add_numbers(a, b):
    result = a + b
    return result


def process_data(items):
    output = []
    for item in items:
        if item > 0:
            output.append(item * 2)
    return output
