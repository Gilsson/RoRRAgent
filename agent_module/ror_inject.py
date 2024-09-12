import struct
import pyMeow as pm  # type: ignore


def read_offsets(proc, base_addr, offsets):
    basepoint = pm.r_int64(proc, base_addr)
    current = basepoint
    for i in offsets[:-1]:
        current = pm.r_int64(proc, current + i)
    return current + offsets[-1]


process = pm.open_process("Risk of Rain Returns.exe")
modules = pm.enum_modules(process)
modules = [module for module in modules if module["name"] == "Risk of Rain Returns.exe"]
# print(modules)
time_base_address = modules[0]["base"] + 0x0218F2E8
# base_address = modules[0]["base"] + 0x01F570B0
time_offsets = [0x340, 0x20, 0x98, 0x48, 0x10, 0x450, 0x0]
# time_offsets = [0x0, 0x10, 0x48, 0x10, 0x270, 0x0]


# addr = read_offsets(process, base_address, offsets)
# print(pm.r_float64(process, pm.pointer_chain_64(process, base_address, time_offsets)))

money_base_address = modules[0]["base"] + 0x01E1BB18
# money_offsets = [0x98, 0x88, 0x70, 0x38, 0x48, 0x10, 0x1B0, 0x0]
money_offsets = [0x98, 0x88, 0x78, 0x38, 0x48, 0x10, 0x1B0, 0x0]
# base_address = modules[0]["base"] + 0x01E1BB18

# print(pm.r_float64(process, pm.pointer_chain_64(process, base_address, money_offsets)))

health_base_address = modules[0]["base"] + 0x021729A0
health_offsets = [0x1D0, 0x18, 0x88, 0x70, 0x38, 0x48, 0x10, 0xC10, 0x0]
# health_offsets = [0x1588, 0x218, 0x10, 0x268, 0x10, 0x48, 0x10, 0xC10, 0x0]

max_health_base_address = modules[0]["base"] + 0x01E1BB18
max_health_offsets = [0x98, 0x88, 0x110, 0x178, 0x38, 0x48, 0x10, 0xD0, 0x0]

# # addr = read_offsets(process, base_address, offsets)
# print(pm.r_float64(process, pm.pointer_chain_64(process, base_address, health_offsets)))


class RoRInject:
    def __init__(self):
        self.health = 0
        self.time = 0
        self.money = 0
        self.max_health = 1
        self.update()

    def update(self):
        try:
            self.health = pm.r_float64(
                process,
                pm.pointer_chain_64(process, health_base_address, health_offsets),
            )
        except Exception as e:
            self.health = 0
        try:
            self.money = pm.r_float64(
                process, pm.pointer_chain_64(process, money_base_address, money_offsets)
            )
        except Exception as e:
            self.money = 0
        try:
            self.time = pm.r_float64(
                process, pm.pointer_chain_64(process, time_base_address, time_offsets)
            )
        except Exception as e:
            self.time = 0
        try:
            self.max_health = pm.r_float64(
                process,
                pm.pointer_chain_64(
                    process, max_health_base_address, max_health_offsets
                ),
            )
        except Exception as e:
            self.max_health = 1


print(RoRInject().max_health)
