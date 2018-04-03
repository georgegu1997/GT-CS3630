import threading
from tkinter import *


class Visualizer():
    """Visualizer to display status of an associated CozMap instance
    """

    def __init__(self, cmap):
        self.cmap = cmap
        self.running = threading.Event()

    def draw_cmap(self):
        """Draw cmap lines
        """
        self.canvas.create_rectangle(0, 0, self.cmap.width, self.cmap.height)

    def draw_color_square(self, coord, color, size=20, bg=False, tags=''):
        """Draw a colored square centered at a given coord

            Arguments:
            coord -- coordinates of square
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            size -- size, in pixels
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords = (coord[0], (self.cmap.height - 1 - coord[1]))
        rect = self.canvas.create_rectangle(coords[0] - size / 2, coords[1] - size / 2, coords[0] + size / 2,
                                            coords[1] + size / 2,
                                            fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_color_circle(self, coord, color, size=5, bg=False, tags=''):
        """Draw a colored circle centered at a given coord

            Arguments:
            coord -- coordinates of square
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            size -- size, in pixels
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords = (coord[0], (self.cmap.height - 1 - coord[1]))
        rect = self.canvas.create_oval(coords[0] - size / 2.0, coords[1] - size / 2.0, coords[0] + size / 2,
                                       coords[1] + size / 2,
                                       fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_color_poly(self, coords, color, bg=False, tags=''):
        """Draw a colored polygon at a given coord

            Arguments:
            coords -- coordinates of vertices
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords_flipped = [(coord[0], (self.cmap.height - 1 - coord[1])) for coord in coords]
        rect = self.canvas.create_polygon(coords_flipped, fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_edge(self, start, end, color, width=1.5, tags=''):
        """Draw an edge segment between two cells

            Arguments:
            start -- starting coordinate
            end -- end coordinate
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            width -- desired width of edge
            tags -- tags to apply to edge
        """
        startcoords = ((start[0] + 0.5), (self.cmap.height - (start[1] + 0.5)))
        endcoords = ((end[0] + 0.5), (self.cmap.height - (end[1] + 0.5)))
        self.canvas.create_line(startcoords[0], startcoords[1], endcoords[0], endcoords[1], fill=color, width=width,
                                arrow=LAST, tags=tags)

    def draw_start(self):
        """Redraw start square
            Color is green by default
        """
        self.canvas.delete('start')
        if self.cmap._start != None:
            self.draw_color_circle(self.cmap._start, '#00DD00', size=15, bg=True, tags='start')

    def draw_goals(self):
        """Redraw all goal cells
            Color is blue by default
        """
        self.canvas.delete('goal')
        for goal in self.cmap._goals:
            self.draw_color_circle(goal, '#0000DD', size=15, bg=True, tags='goal')

    def draw_obstacles(self):
        """Redraw all obstacles
            Color is dark gray by default
        """
        self.canvas.delete('obstacle')
        for obstacle in self.cmap._obstacles:
            self.draw_color_poly(obstacle, '#222222', bg=True, tags='obstacle')

    def draw_nodes(self):
        """"Redraw all nodes, these nodes are in RRT
        """
        self.canvas.delete('nodes')
        for node in self.cmap._nodes:
            self.draw_color_circle(node, '#CCCCCC', bg=True, tags='nodes')

    def draw_node_path(self):
        """"Redraw all node paths
        """
        self.canvas.delete('node_paths')
        for node_path in self.cmap._node_paths:
            self.draw_edge(node_path[0], node_path[1], color='#DD0000', width=2, tags='node_paths')

    def draw_solution(self):
        """"Redraw one solution from start to goal
        """
        self.canvas.delete('solved')
        path = self.cmap.get_smooth_path()
        for p in range(len(path)-1):
            self.draw_edge(path[p], path[p+1], color='#DDDD00', width=5, tags='solved')

    def update(self, *args):
        """Redraw any updated cmap elements
        """

        self.cmap.lock.acquire()
        self.running.clear()
        self.cmap.updated.clear()

        if 'start' in self.cmap.changes:
            self.draw_start()
        if 'goals' in self.cmap.changes:
            self.draw_goals()
        if 'obstacles' in self.cmap.changes:
            self.draw_obstacles()
        if 'nodes' in self.cmap.changes:
            self.draw_nodes()
        if 'node_paths' in self.cmap.changes:
            self.draw_node_path()
        if 'solved' in self.cmap.changes:
            self.draw_solution()

        self.cmap.changes = []
        self.running.set()
        self.cmap.lock.release()

    def setup(self):
        """Do initial drawing of cmap, start, goals, and obstacles
        """
        self.cmap.lock.acquire()
        self.draw_cmap()
        self.draw_start()
        self.draw_goals()
        self.draw_obstacles()
        self.cmap.lock.release()

    def start(self):
        """Start the visualizer, must be done in main thread to avoid issues on macs
            Blocks until spawned window is closed
        """
        # Set up Tk stuff
        master = Tk()
        master.title("CS 3630 Lab 4 RRT")
        self.canvas = Canvas(master, width=self.cmap.width, height=self.cmap.height, bd=0, bg='#FFFFFF')
        self.canvas.pack()

        # Draw cmap and any initial items
        self.setup()

        # Start mainloop and indicate that it is running
        self.running.set()

        while True:
            if self.cmap.updated.is_set():
                self.update()
            try:
                master.update_idletasks()
                master.update()
            except TclError:
                break

        # Indicate that main loop has finished
        self.running.clear()
