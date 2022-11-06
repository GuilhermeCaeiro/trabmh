from select import select


class Selection:
    def __init__(self):
        pass

    @classmethod
    def select(self, individuals, function_name):
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(individuals)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

    @classmethod
    def select_parents(self, individuals, method):
        function_name = "parents" + "_" + method
        return select(individuals, function_name)
        
    @classmethod
    def select_nextgen(self, individuals, method):
        function_name = "nextgen" + "_" + method
        return select(individuals, function_name)