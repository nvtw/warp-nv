# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests the CUDA collision driver."""

import engine_collision_driver
import jax
import mujoco
import numpy as np
from jax import numpy as jp
from mujoco import mjx

import warp as wp
import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union



def vmap(fn: Callable, in_axes: int | None | Sequence[Any] = 0):
    batch_dim = None

    def squeeze_array(a: jax.Array) -> jax.Array:
        # remove batch dimension
        return a.reshape(-1, *a.shape[2:])

    def unsqueeze_array(a: Union[np.ndarray, jax.Array]) -> jax.Array:
        # add batch dimension
        shape_div_batch = a.shape[0] // batch_dim
        new_shape = (batch_dim, shape_div_batch, *a.shape[1:])
        return a.reshape(*new_shape)

    def vec_fn(*args):
        nonlocal batch_dim
        squeezed_args = []
        for i_arg, arg in enumerate(args):
            if in_axes is not None:
                if isinstance(in_axes, int):
                    if i_arg != in_axes:
                        squeezed_args.append(arg)
                        continue
                if i_arg < len(in_axes) and in_axes[i_arg] is None:
                    squeezed_args.append(arg)
                    continue
            if isinstance(arg, mujoco.mjx._src.types.PyTreeNode):
                arg_cpy = copy.copy(arg)
                for key, value in arg.__dict__.items():
                    if isinstance(value, jax.Array):
                        if batch_dim is None:
                            batch_dim = value.shape[0]
                        if value.shape[0] != batch_dim:
                            raise ValueError("All arrays must have the same batch dimension")
                        arg_cpy.__dict__[key] = squeeze_array(value)
                squeezed_args.append(arg_cpy)
            elif isinstance(arg, jax.Array):
                squeezed_args.append(squeeze_array(arg))
            else:
                squeezed_args.append(arg)
        with wp.ScopedTimer(f"vmap_{fn.__name__}", use_nvtx=True):
            results = fn(*squeezed_args)
        batched_results = []
        for result in results:
            if isinstance(result, np.ndarray) or isinstance(result, jax.Array):
                batched_results.append(unsqueeze_array(result))
            else:
                batched_results.append(result)
        return batched_results

    return vec_fn



def compare_contacts2(test_cls, jax_contacts, wp_contacts):
    """Compare jax contacts and warp contacts. The first dimension is the batch."""
    for env_id, (g1, g2) in enumerate(zip(jax_contacts["geom"], wp_contacts["geom"])):
        for g1_key in np.unique(g1, axis=0):
            idx1 = np.where((g1 == g1_key).all(axis=1))
            idx2 = np.where((g2 == g1_key).all(axis=1))
            test_cls.assertTrue(len(idx1) > 0)
            dist1 = jax_contacts["dist"][env_id][idx1]
            dist2 = wp_contacts["dist"][env_id][idx2]
            # contacts may appear in JAX with dist>0, but not in CUDA.
            # (not sure about this part)
            if (dist1 > 0).any():
                if dist2.shape[0]:
                    test_cls.assertTrue((dist2 >= 0).any())
                continue
            test_cls.assertTrue((dist1 < 0).all())
            # contact distance in JAX are dynamically calculated, so we only
            # check that CUDA distances are equal to the first JAX distance.
            np.testing.assert_array_almost_equal(dist1[0], dist2, decimal=3)
            # normals should be equal.
            shape1_debugA = jax_contacts["frame"].shape
            shape2_debugA = wp_contacts["frame"].shape
            shape1_debug = jax_contacts["frame"][env_id, :, 0].shape
            shape2_debug = wp_contacts["frame"][env_id, :, 0].shape

            normal1 = jax_contacts["frame"][env_id, :, 0][idx1]
            normal2 = wp_contacts["frame"][env_id, :, 0][idx2]
            test_cls.assertLess(np.abs(normal1[0] - normal2).max(), 1e-5)
            # contact points are not as accurate in CUDA, the test is rather loose.
            found_point = 0
            pos1 = jax_contacts["pos"][env_id][idx1]
            pos2 = wp_contacts["pos"][env_id][idx2]
            for pos1_idx in range(pos1.shape[0]):
                pos2_idx = np.abs(pos1[pos1_idx] - pos2).sum(axis=1).argmin()
                found_point += np.abs(pos1[pos1_idx] - pos2[pos2_idx]).max() < 0.11
            test_cls.assertGreater(found_point, 0)


def _compare_contacts(test_cls, dx, c):
    """Compares JAX and CUDA contacts."""
    for env_id, (g1, g2) in enumerate(zip(dx.contact.geom, c.geom)):
        for g1_key in np.unique(g1, axis=0):
            idx1 = np.where((g1 == g1_key).all(axis=1))
            idx2 = np.where((g2 == g1_key).all(axis=1))
            dist1 = dx.contact.dist[env_id][idx1]
            dist2 = c.dist[env_id][idx2]
            # contacts may appear in JAX with dist>0, but not in CUDA.
            if (dist1 > 0).any():
                if dist2.shape[0]:
                    test_cls.assertTrue((dist2 >= 0).any())
                continue
            test_cls.assertTrue((dist1 < 0).all())
            # contact distance in JAX are dynamically calculated, so we only
            # check that CUDA distances are equal to the first JAX distance.
            np.testing.assert_array_almost_equal(dist1[0], dist2, decimal=3)
            # normals should be equal.
            normal1 = dx.contact.frame[env_id, :, 0][idx1]
            normal2 = c.frame[env_id, :, 0][idx2]
            test_cls.assertLess(np.abs(normal1[0] - normal2).max(), 1e-5)
            # contact points are not as accurate in CUDA, the test is rather loose.
            found_point = 0
            pos1 = dx.contact.pos[env_id][idx1]
            pos2 = c.pos[env_id][idx2]
            for pos1_idx in range(pos1.shape[0]):
                pos2_idx = np.abs(pos1[pos1_idx] - pos2).sum(axis=1).argmin()
                found_point += np.abs(pos1[pos1_idx] - pos2[pos2_idx]).max() < 0.11
            test_cls.assertGreater(found_point, 0)


class EngineCollisionDriverTest:  # (absltest.TestCase):
    _CONVEX_CONVEX = """
    <mujoco>
      <asset>
        <mesh name="meshbox"
              vertex="-1 -1 -1
                      1 -1 -1
                      1  1 -1
                      1  1  1
                      1 -1  1
                      -1  1 -1
                      -1  1  1
                      -1 -1  1"/>
        <mesh name="poly" scale="0.5 0.5 0.5"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
        <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
          vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
          face="0 1 2  0 3 1  0 2 3  1 3 2"/>
      </asset>
      <custom>
        <numeric data="12" name="max_contact_points"/>
      </custom>
      <worldbody>
        <light pos="-.5 .7 1.5" cutoff="55"/>
        <body pos="0.0 2.0 0.35" euler="0 0 90">
          <freejoint/>
          <geom type="mesh" mesh="meshbox"/>
        </body>
        <body pos="0.0 2.0 1.781" euler="180 0 0">
          <freejoint/>
          <geom type="mesh" mesh="poly"/>
          <geom pos="0.5 0 -0.2" type="sphere" size="0.3"/>
        </body>
        <body pos="0.0 2.0 2.081">
          <freejoint/>
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
        <body pos="0.0 0.0 -2.0">
          <geom type="plane" size="40 40 40"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def assertTupleEqual(self, actual, expected, msg=None):
        """Checks if two tuples are equal and raises an AssertionError if not."""
        if actual != expected:
            error_message = f"Tuples are not equal: {actual} != {expected}"
            if msg:
                error_message = f"{msg}\n{error_message}"
            raise AssertionError(error_message)

    def test_shapes(self):
        """Tests collision driver return shapes."""
        m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        batch_size = 12

        #@jax.vmap
        def make_model_and_data(val):
            dx = mjx.make_data(m)
            mx = mjx.put_model(m)
            dx = dx.replace(qpos=dx.qpos.at[2].set(val))
            return mx, dx       

        # # vary the size of body 0.
        # mx, dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

        # kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics))
        # dx = kinematics_jit_fn(mx, dx)

        # vec_collision = vmap(
        #     engine_collision_driver.collision2,
        #          in_axes = (
        #             0,
        #             0,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,                     
        #              ),
        #           )
        # c = vec_collision(mx, dx, 1e9, 12, 12, 12, 8, 1.0)


        # return
    

        mx_list = []
        dx_list = []
        for val in jp.arange(-1, 1, 2 / batch_size):
            mx, dx = make_model_and_data(val)
            mx_list.append(mx)
            dx_list.append(dx)
      
        forward_jit_fn = jax.jit(mjx.forward)


        for i in range(batch_size):
            dx_list[i] = forward_jit_fn(mx_list[i], dx_list[i])

        c_list = []
        for i in range(batch_size):
            c = engine_collision_driver.collision2(mx_list[i], dx_list[i], 1e9, 12, 12, 12, 8, 1.0, i)
            c_list.append(c)


        dx_batched = {
            field: np.stack([getattr(obj.contact, field) for obj in dx_list]) for field in ("frame", "dist", "geom", "pos", "friction", "solimp", "solref", "solreffriction", "geom1", "geom2")
        }

        c_batched = {
            field: np.stack([getattr(obj, field) for obj in c_list]) for field in ("frame", "dist", "geom", "pos", "friction", "solimp", "solref", "solreffriction", "geom1", "geom2")
        }	



        npts = dx_batched["pos"].shape[1]
        self.assertTupleEqual(c_batched["dist"].shape, (batch_size, npts))
        self.assertTupleEqual(c_batched["pos"].shape, (batch_size, npts, 3))
        self.assertTupleEqual(c_batched["frame"].shape, (batch_size, npts, 3, 3))
        self.assertTupleEqual(c_batched["friction"].shape, (batch_size, npts, 5))
        self.assertTupleEqual(c_batched["solimp"].shape, (batch_size, npts, mujoco.mjNIMP))
        self.assertTupleEqual(c_batched["solref"].shape, (batch_size, npts, mujoco.mjNREF))
        self.assertTupleEqual(c_batched["solreffriction"].shape, (batch_size, npts, mujoco.mjNREF))
        self.assertTupleEqual(c_batched["geom"].shape, (batch_size, npts, 2))
        self.assertTupleEqual(c_batched["geom1"].shape, (batch_size, npts))
        self.assertTupleEqual(c_batched["geom2"].shape, (batch_size, npts))


    def test_contacts_batched_model_data(self):
        """Tests collision driver results."""
        m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        batch_size = 3

        #@jax.vmap
        def make_model_and_data(val):
            dx = mjx.make_data(m)
            dx = dx.replace(qpos=dx.qpos.at[2].set(val))
            mx = mjx.put_model(m)
            return mx, dx

        # vary the z-position of body 0.
        #mx, dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

        mx_list = []
        dx_list = []
        for val in jp.arange(-1, 1, 2 / batch_size):
            mx, dx = make_model_and_data(val)
            mx_list.append(mx)
            dx_list.append(dx)
      
        forward_jit_fn = jax.jit(mjx.forward)


        for i in range(batch_size):
            dx_list[i] = forward_jit_fn(mx_list[i], dx_list[i])

        c_list = []
        for i in range(batch_size):
            c = engine_collision_driver.collision2(mx_list[i], dx_list[i], 1e9, 12, 12, 12, 8, 1.0, i)
            c_list.append(c)


        dx_batched = {
            field: np.stack([getattr(obj.contact, field) for obj in dx_list]) for field in ("frame", "dist", "geom", "pos")
        }

        c_batched = {
            field: np.stack([getattr(obj, field) for obj in c_list]) for field in ("frame", "dist", "geom", "pos")
        }	
        for i in range(batch_size):
            print(i)
            print(dx_list[i].contact.geom)
            print(c_list[i].geom)

        # test contact normals and penetration depths.
        # workaround with simple assert function
        class Assert:
            def assertTrue(cond):
                assert cond

            def assertGreater(cond, ref):
                assert cond > ref

            def assertLess(cond, ref):
                assert cond < ref

        compare_contacts2(Assert, dx_batched, c_batched)



    def test_contacts_batched_data(self):
        """Tests collision driver results."""
        m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        batch_size = 3

        def make_model_and_data(val):
            dx = mjx.make_data(m)
            mx = mjx.put_model(m)
            dx = dx.replace(qpos=dx.qpos.at[2].set(val))
            return mx, dx

        mx_list = []
        dx_list = []
        for val in jp.arange(-1, 1, 2 / batch_size):
            mx, dx = make_model_and_data(val)
            mx_list.append(mx)
            dx_list.append(dx)

        # vary the z-position of body 0.
        # mx = mjx.put_model(m)
        # dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

        forward_jit_fn = jax.jit(mjx.forward)

        for i in range(batch_size):
            dx_list[i] = forward_jit_fn(mx_list[i], dx_list[i])

        c_list = []
        for i in range(batch_size):
            c = engine_collision_driver.collision2(mx_list[i], dx_list[i], 1e9, 12, 12, 12, 8, 1.0)
            c_list.append(c)

        # Initialize an empty dictionary to hold the batched data
        dx_batched = {}

        # Iterate over the fields of interest
        for field in ("frame", "dist", "geom", "pos"):
            # Create an empty list to store the individual arrays for this field
            field_values = []
            
            # Iterate over the objects in dx_list and get the contact attribute for the current field
            for obj in dx_list:
                val = getattr(obj.contact, field)
                field_values.append(val)
            
            # Stack the list of arrays for the current field and add it to the dictionary
            dx_batched[field] = np.stack(field_values)



        # Initialize an empty dictionary to hold the batched data
        c_batched = {}

        # Iterate over the fields of interest
        for field in ("frame", "dist", "geom", "pos"):
            # Create an empty list to store the individual arrays for this field
            field_values = []
            
            # Iterate over the objects in c_list and get the attribute for the current field
            for obj in c_list:
                val = getattr(obj, field)
                field_values.append(val)
            
            # Stack the list of arrays for the current field and add it to the dictionary
            c_batched[field] = np.stack(field_values)
       

        #shapeA = dx_batched.shape
        #shapeB = c_batched.shape

        # workaround with simple assert function
        class Assert:
            def assertTrue(cond):
                assert cond

            def assertGreater(cond, ref):
                assert cond > ref

            def assertLess(cond, ref):
                assert cond < ref

        compare_contacts2(Assert, dx_batched, c_batched)



    _CONVEX_CONVEX_2 = """
    <mujoco>
      <asset>
        <mesh name="poly" scale="0.5 0.5 0.5"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
        <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
          vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
          face="0 1 2  0 3 1  0 2 3  1 3 2"/>
      </asset>
      <custom>
        <numeric data="2" name="max_contact_points"/>
      </custom>
      <worldbody>
        <light pos="-.5 .7 1.5" cutoff="55"/>
        <body pos="0.0 2.0 1.781" euler="180 0 0">
          <freejoint/>
          <geom type="mesh" mesh="poly"/>
        </body>
        <body pos="0.0 2.0 2.081">
          <freejoint/>
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def test_solparams(self):
        """Tests collision driver solparams."""
        m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX_2)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        batch_size = 3

        #@jax.vmap
        def make_model_and_data(val):
            dx = mjx.make_data(m)
            mx = mjx.put_model(m)
            mx = mx.replace(
                geom_solref=mx.geom_solref.at[0].set(val),
                geom_solimp=mx.geom_solimp.at[0].set(val),
                geom_friction=mx.geom_friction.at[0].set(val),
            )
            return mx, dx

        # vary geom contact solver params
        # mx, dx = make_model_and_data(jp.arange(0.1, 1.1, 1 / batch_size))
		
        mx_list = []
        dx_list = []
        for val in jp.arange(0.1, 1.1, 1 / batch_size):
            mx, dx = make_model_and_data(val)
            mx_list.append(mx)
            dx_list.append(dx)

        forward_jit_fn = jax.jit(mjx.forward)

        for i in range(batch_size):
            dx_list[i] = forward_jit_fn(mx_list[i], dx_list[i])

        c_list = []
        for i in range(batch_size):
            c = engine_collision_driver.collision2(mx_list[i], dx_list[i], 1e9, 12, 12, 12, 8, 1.0)
            c_list.append(c)
			
        dx_batched = {
            field: np.stack([getattr(obj.contact, field) for obj in dx_list]) for field in ("solref", "solimp", "friction")
        }

        c_batched = {
            field: np.stack([getattr(obj, field) for obj in c_list]) for field in ("solref", "solimp", "friction")
        }	


        np.testing.assert_array_almost_equal(c_batched["solref"], dx_batched["solref"])
        np.testing.assert_array_almost_equal(c_batched["solimp"], dx_batched["solimp"])
        np.testing.assert_array_almost_equal(c_batched["friction"], dx_batched["friction"])



if __name__ == "__main__":


    jax.config.update("jax_platform_name", "cpu")
    # https://nvidia.github.io/warp/debugging.html
    wp.init()

    wp.set_device("cpu")
    wp.config.mode = "debug"
    # assert wp.context.runtime.core.is_debug_enabled(), "Warp must be built in debug mode to enable debugging kernels"

    wp.config.verify_fp = True
    wp.config.print_launches = True
    wp.config.verify_cuda = True

    instance = EngineCollisionDriverTest()
    # instance.test_shapes()
    instance.test_contacts_batched_model_data()
    #instance.test_contacts_batched_data()
    #instance.test_solparams()
   
    print("done")

    # absltest.main()
