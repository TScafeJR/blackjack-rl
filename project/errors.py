class Errors:
    @staticmethod
    def value_undefined(self, val: str):
        raise Exception(f"Value {val} is undefined")
