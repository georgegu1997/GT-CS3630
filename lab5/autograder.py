from rrt import *


class GradingThread():
    """RRT grading thread
    """
    def __init__(self, maps):
        self.maps = maps

    def run(self):
        print("Grader running...\n")
        points = 0
        total = len(self.maps) * 10

        for map in self.maps:
            cmap = CozMap(map, node_generator)
            RRT(cmap, cmap.get_start())
            if (cmap.is_solution_valid()):
                points += 10
            print(map + ": " + str(points) + "/" + str(total) + " points")

        print("\nScore = " + str(points) + "/" + str(total) + "\n")


if __name__ == "__main__":
    tests = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1]) as testfile:
                tests = json.loads(testfile.read())
        except:
            print("Error opening test file, please check filename and json format")
            raise
    else:
        print("correct usage: python3 autograder.py <testfile>")
        exit()

    grader = GradingThread(tests["mapfile"])
    grader.run()
