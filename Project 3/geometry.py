from room import Room

class Apartment:
    def __init__(self, layout="default", dx=1/20):
        if layout=="default":
            omega1=Room(dx, (1, 1), ["left"], [])
            omega2=Room(dx, (1, 2), ["top"], ["bottom"])
            omega3=Room(dx, (1, 1), ["right"], [])
            coupling1_2={"neighbor": omega2, "side":"right", "start":0.0, "end":1.0}
            coupling2_1={"neighbor": omega1, "side": "left", "start": 0.0, "end": 1.0}
            coupling2_3={"neighbor": omega3, "side": "right", "start": 1.0, "end": 2.0}
            coupling3_2={"neighbor": omega2, "side": "left", "start": 0.0, "end": 1.0}
            omega1.add_coupling(coupling1_2)
            omega2.add_coupling(coupling2_1)
            omega2.add_coupling(coupling2_3)
            omega3.add_coupling(coupling3_2)
            self.rooms=[omega1, omega2, omega3]
        else:
            raise NotImplementedError("only default layout implemented")
    def iterate(self):
        return None #not implemented
