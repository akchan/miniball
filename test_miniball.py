#
# Copyright (c) 2019-2023 Alexandre Devert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy
import miniball


def test_repeatability():
    # Check that we can have repeatable results when providing the RNG
    rng_seed = 42

    S = numpy.random.randn(100, 2)
    C_a, r2_a = miniball.get_bounding_ball(
        S, rng=numpy.random.default_rng(seed=rng_seed)
    )
    C_b, r2_b = miniball.get_bounding_ball(
        S, rng=numpy.random.default_rng(seed=rng_seed)
    )

    assert (C_a == C_b).all()
    assert (r2_a == r2_b).all()


def test_integer_coordinates():
    # Check that integer coordinates are properly handled
    for n in range(1, 10):
        S_int = numpy.random.randint(-1000, 1500, (100, 2))
        C_int, r2_int = miniball.get_bounding_ball(S_int)

        S = S_int.astype(float)
        C, r2 = miniball.get_bounding_ball(S)

        assert numpy.allclose(C, C_int)
        assert numpy.allclose(r2, r2_int)


def test_bounding_ball_contains_point_set():
    # Check that the computed bounding ball contains all the input points
    for n in range(1, 10):
        for count in range(2, n + 10):
            # Generate points
            S = numpy.random.randn(count, n)

            # Get the bounding sphere
            C, r2 = miniball.get_bounding_ball(S)

            # Check that all points are inside the bounding sphere up to
            # machine precision
            assert numpy.all(
                numpy.square(S - C).sum(axis=1) - r2 < 1e-12
            )


def test_bounding_ball_optimality():
    # Check that the bounding ball are optimal
    for n in range(2, 10):
        for count in range(n + 2, n + 30):
            # Generate a support sphere from n+1 points
            S_support = numpy.random.randn(n + 1, n)
            C_support, r2_support = miniball.get_bounding_ball(S_support)

            # Generate points inside the support sphere
            S = numpy.random.randn(count - S_support.shape[0], n)
            S /= numpy.sqrt(numpy.square(S).sum(axis=1))[:, None]
            S *= (0.9 * numpy.sqrt(r2_support)) * numpy.random.rand(
                count - S_support.shape[0], 1
            )
            S = S + C_support

            # Get the bounding sphere
            C, r2 = miniball.get_bounding_ball(
                numpy.concatenate([S, S_support], axis=0)
            )

            # Check that the bounding sphere and the support sphere are
            # equivalent up to machine precision.
            assert numpy.allclose(r2, r2_support)
            assert numpy.allclose(C, C_support)


def test_bounding_ball_3D_coplanar_cases():
    coplanar_points = [
        [numpy.array([[0.025728409071614222, 0.372152791836762, 0.3313301933484486],
                      [0.2531163257212625, 0.9836838066044385, 0.2053900271644188],
                      [0.01222939518218047, 0.8318221612473629, 0.6333422846064242],
                      [0.15230829793039352, 0.45195218214902, 0.10645228460244449]
                      ]),
         [numpy.array([0.114472, 0.70108076, 0.33578222]), 0.11608885468045413]],
        [numpy.array([[0.019703769768801283, 0.22959782389636796, 0.9347328105384431],
                      [0.025728409071614222, 0.372152791836762, 0.3313301933484486],
                      [0.07256394493759732, 0.18814654709738765, 0.6659776158852926],
                      [-0.28448686909485404, 0.9845638879736157, 0.4729161039578733],
                      [-0.29051150839766654, 0.8420089200332219, 1.0763187211478678],
                      [-0.3373470442636498, 1.0260151647725961, 0.7416712986110239]
                      ]),
            [numpy.array([-0.13239155,  0.60708086,  0.70382446]), 0.2189450932893729]],
        [numpy.array([[0.6230334535523369, 0.10461020463586657, 0.6246333540949156],
                      [0.4368180091809194, 0.47867605481000886, 0.4940056593092361],
                      [0.19956709139681328, 0.3156469527895308, 0.46603709980425945],
                      [0.19258810832074413, 0.00570894472836192, 0.77173157151973],
                      [0.07256394493759732, 0.18814654709738765, 0.6659776158852926],
                      [0.14079426383352756, 0.46309890200523474, 0.8713935666594159],
                      [0.3270097082049448, 0.08903305183109245, 1.0020212614450952],
                      [0.22415372406335649, -0.014249183710294266, 0.7522170325705094],
                      [0.2311327071394254, 0.29568882435087457, 0.4465225608550386],
                      [0.2954212758833918, 0.06960900045054658, 0.506756030291579]
                      ]),
            [numpy.array([0.37123437, 0.27549227, 0.73929129]), 0.10574990180985341]],
        [numpy.array([[165.9375, 90.9375, 39.19999933],
                      [169.6875, 87.1875, 39.19999933],
                      [165.9375, 87.1875, 39.19999933],
                      [169.6875, 90.9375, 39.19999933]]),
            [numpy.array([167.8125, 89.0625, 39.19999933]), 7.031250000000002]],
    ]

    for input, expected in coplanar_points:
        C, r2 = miniball.get_bounding_ball(input)
        assert numpy.allclose(C, expected[0])
        assert numpy.allclose(r2, expected[1])
