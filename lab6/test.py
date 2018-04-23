import cozmo, time
from cozmo.util import degrees, distance_mm, speed_mmps



async def cozmo_program(robot: cozmo.robot.Robot):
    while True:
        if robot.is_picked_up:
            print("is_picked_up")
            continue

        # if time.time() - start < 3:
        #     robot.drive_wheels(50.0, 50.0, duration = 3.0)
        # else:
        #     robot.drive_wheels(0.0, 0.0)


cozmo.run_program(cozmo_program)
