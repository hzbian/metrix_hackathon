from collections import OrderedDict

PLANES = ["U41_318eV",
          "ASBL",
          "M1-Cylinder",
          "Spherical Grating",
          "Exit Slit",
          "E1",
          "E2",
          "ImagePlane"]

PARAMS_INFO_SEPARATE = {
    "U41_318eV": [('U41_318eV.numberRays', (1e5, 1e5)),
                  ('U41_318eV.translationXerror', (-0.25, 0.25)),
                  ('U41_318eV.translationYerror', (-0.25, 0.25)),
                  ('U41_318eV.rotationXerror', (-0.05, 0.05)),
                  ('U41_318eV.rotationYerror', (-0.05, 0.05))],
    "ASBL": [('ASBL.totalWidth', (1.9, 2.1)),
             ('ASBL.totalHeight', (0.9, 1.1)),
             ('ASBL.translationXerror', (-0.2, 0.2)),
             ('ASBL.translationYerror', (-0.2, 0.2))],
    "M1-Cylinder": [('M1_Cylinder.radius', (174.06, 174.36)),
                    ('M1_Cylinder.rotationXerror', (-0.25, 0.25)),
                    ('M1_Cylinder.rotationYerror', (-1., 1.)),
                    ('M1_Cylinder.rotationZerror', (-1., 1.)),
                    ('M1_Cylinder.translationXerror', (-0.15, 0.15)),
                    ('M1_Cylinder.translationYerror', (-1., 1.))],
    "Spherical Grating": [('SphericalGrating.radius', (109741., 109841.)),
                          ('SphericalGrating.rotationYerror', (-1., 1.)),
                          ('SphericalGrating.rotationZerror', (-2.5, 2.5))],
    "Exit Slit": [('ExitSlit.totalHeight', (0.009, 0.011)),
                  ('ExitSlit.translationZerror', (-29., 31.)),
                  ('ExitSlit.rotationZerror', (-0.3, 0.3))],
    "E1": [('E1.longHalfAxisA', (20600., 20900.)),
           ('E1.shortHalfAxisB', (300.721702601, 304.721702601)),
           ('E1.rotationXerror', (-0.5, 0.5)),
           ('E1.rotationYerror', (-7.5, 7.5)),
           ('E1.rotationZerror', (-4, 4)),
           ('E1.translationYerror', (-1, 1)),
           ('E1.translationZerror', (-1, 1))],
    "E2": [('E2.longHalfAxisA', (4325., 4425.)),
           ('E2.shortHalfAxisB', (96.1560870104, 98.1560870104)),
           ('E2.rotationXerror', (-0.5, 0.5)),
           ('E2.rotationYerror', (-7.5, 7.5)),
           ('E2.rotationZerror', (-4, 4)),
           ('E2.translationYerror', (-1, 1)),
           ('E2.translationZerror', (-1, 1))],
    "ImagePlane": []
}

PLANES_INFO = {}
for idx, plane in enumerate(PLANES):
    PARAM_NAMES = []
    for plane_ in PLANES[:idx + 1]:
        PARAM_NAMES += [name for name, _ in PARAMS_INFO_SEPARATE[plane_]]
    PLANES_INFO[plane] = ([f"1e5/ray_output/{plane}/hist_small"], PARAM_NAMES)

PARAMS_KEY = '1e5/params'

PARAMS_INFO = {}
for v in PARAMS_INFO_SEPARATE.values():
    PARAMS_INFO.update(OrderedDict(v))
