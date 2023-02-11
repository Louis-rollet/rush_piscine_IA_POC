
# We need to load the common voice database
class DataBase:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.load()

