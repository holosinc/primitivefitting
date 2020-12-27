class UnityCollider:
    pass

class UnityCube(UnityCollider):
    def __init__(self, position, rotation, size):
        self.position = position
        self.rotation = rotation
        self.size = size

    def __str__(self):
        return "Cube\nPos: " + str(self.position) + "\nRot: " + str(self.rotation) + "\nSize: " + str(self.size)

class UnityCapsule(UnityCollider):
    def __init__(self, position, rotation, height, radius):
        self.position = position
        self.rotation = rotation
        self.height = height
        self.radius = radius

    def __str__(self):
        return "Capsule (x-axis direction)\nPos: " + str(self.position) + "\nRot: " + str(self.rotation) + "\nHeight: " + str(self.height) + "\nRadius: " + str(self.radius)

class UnitySphere(UnityCollider):
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def __str__(self):
        return "Sphere\nPos: " + str(self.position) + "\nRadius: " + str(self.radius)
