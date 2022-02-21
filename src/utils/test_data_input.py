import DataInterface

manager = DataInterface.DeviceManager()
on_box = manager.on_box_device
print(on_box.data())