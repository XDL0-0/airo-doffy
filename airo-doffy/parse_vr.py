import re

def parse_data(input_data):
    if input_data is None:
        return None
    
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, input_data.strip())
    controllers = []
    for match in matches:
        controller_data = {}
        key_value_pairs = re.findall(r"(\w+)\s*=\s*((?:\([^\)]+\)|[^\s,]+))", match)
        for key, value in key_value_pairs:
            if value.startswith("(") and value.endswith(")"):
                value = tuple(map(float, value.strip("()").split(", ")))
            else:
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    value = value
            controller_data[key] = value
        controllers.append(controller_data)

    return controllers

