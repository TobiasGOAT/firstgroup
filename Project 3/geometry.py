import room

class Apartment:
    def __init__(self, layout="default", dx=1/20):
        omega1=room("omega1", dx, (1, 1), ["left"], [])
        omega2=room("omega2", dx, (1, 2), ["top"], ["bottom"])
        omega3=room("omega3", dx, (1, 1), ["right"], [])
        coupling1_2={"neightbor": omega2, "side":"right", "start":0, "end":1.0}
        coupling2_1={"neighbor": omega1, "side": "left", "start": 0, "end": 1.0}
        coupling2_3={"neighbor": omega3, "side": "right", "start": 1.0, "end": 2.0}
        coupling3_2={"neighbor": omega2, "side": "left", "start": 0, "end": 1.0}
        omega1.add_coupling(coupling1_2)
        omega2.add_coupling(coupling2_1)
        omega2.add_coupling(coupling2_3)
        omega3.add_coupling(coupling3_2)
        