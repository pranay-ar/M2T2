    # rr.init("grasp_viz", spawn=True)
    # rr.log("camera/rgb", rr.Image(rgb))
    # rr.log("camera/depth", rr.DepthImage(depth))
    # # rr.log("camera/segmap", rr.SegmentationImage(segmap))
    # rr.log("scene/origin", rr.Transform3D(axis_length=1.0))

    # colors = [0xFF0000FF, 0x00FF00FF, 0x0000FFFF, 0xFFFF00FF, 0x00FFFFFF]
    # radii = [0.003] * 5

    # rr.log("scene/points", rr.Points3D(pcd_vis, colors=colors))
    # for i in range(1):
    #     rr.log(f"scene/points_{i}", rr.Points3D(positions[i], colors=colors[i],  radii=radii[i]))
    #     rr.log(f"scene/contact_points_{i}", rr.Points3D(contact_pts[i], colors=colors[i],  radii=radii[i]))

    #     rr.log(
    #         f"scene/grasp_transform_{i}",
    #         rr.Transform3D(
    #             translation=[positions[i][0], positions[i][1], positions[i][2]],
    #             quaternion=quaternions[i],
    #             axis_length=0.1
    #         )
    #     )

    #     # rr.log(
    #     #     f"scene/robot_grasp_transform_{i}",
    #     #     rr.Transform3D(
    #     #         translation=positions[i],
    #     #         quaternion=quat_ee[i],
    #     #         axis_length=0.1
    #     #     )
    #     # )

    # rr.log(
    #     f"scene/curr_right_{i}",
    #     rr.Transform3D(
    #         translation=[curr_pos_right.x_val, curr_pos_right.y_val, curr_pos_right.z_val],
    #         quaternion=[curr_pos_xyzw.x, curr_pos_xyzw.y, curr_pos_xyzw.z, curr_pos_xyzw.w],
    #         axis_length=0.1
    #     )
    # )