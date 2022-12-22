import os
import subprocess
import argparse
from typing import List


# HOW TO use: python3 script_rayui_bg.py /pfad/test.rml -p TestPlane TestPlane2

def query(command, proc):
    proc.stdin.write(bytes(command + "\n", encoding='utf-8'))
    proc.stdin.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rml_workfile",
                        type=str,
                        help="The RML workfile path")
    parser.add_argument('-p', '--exported_planes',
                        nargs='+',
                        default=['ImagePlane'],
                        help='Exported planes',
                        required=False)
    args = parser.parse_args()
    rml_workfile: str = args.rml_workfile
    exported_planes: List[str] = args.exported_planes
    proc = subprocess.Popen('/opt/run.sh', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    workdir = os.path.dirname(rml_workfile)

    query('load ' + rml_workfile, proc)
    query("trace noanalyze", proc)
    for exported_plane in exported_planes:
        query("export \"" + exported_plane + "\" \"RawRaysBeam\" " + workdir, proc)
    query("quit", proc)
    proc.stdin.close()
    proc.wait()
