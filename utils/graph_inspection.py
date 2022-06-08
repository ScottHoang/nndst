import os
import sys

import networkx

if __name__ == "__main__":
    _, file = sys.argv
    network = networkx.read_gml(file)
    masked = 0
    total = 0
    for e, attr in network.edges.items():
        total += 1
        if attr['mask'] == 1:
            masked += 1
        print(e, attr['mask'])
    print(f"masked: {masked}, total: {total}")
