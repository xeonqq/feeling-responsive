---
layout: page
#
# Content
#
subheadline: "Robots"
title: "Home-made Self-balance Robot"
teaser: "Do you like robots? Or even better, a Segway like self balance robot? I will teach you how to make one yourself"
categories:
  - design
tags:
  - robot
  - arduino
  - pid
  - kalman filter
comments: true
#
# Styling
#
header: no
image:
    title: robot-title.jpg
    thumb: robot-thumb.jpg
    homepage: robot-title.jpg
---

During my free time I built this robot which can balance itself on two wheels and can handle small disturbance. Further work shall be done on remote controlling to move it.

The objective is to have some hands-on with proccessing the data of an inertia sensor, and also implementing the PID controller to have some insight of how PID works. In the end, of course, to have some fun!
  
Code: [Github Repo](https://github.com/xeonqq/balance_robot)

## »Demo
<iframe width="560" height="315" src="https://www.youtube.com/embed/0947fcgWL5s" frameborder="0" allowfullscreen></iframe>


## »Hardwares
1. [Arduino Leonardo Board][1]
2. [2 DC Motors: JGA25-371 with encoders][2]
3. [Arduino Motor Driver Shield][3]
4. [12V-5V-3.3V Power Convertion Board][4]
5. [ZIPPY Flightmax 2200mAh 3S1P 25C Battery][5]
6. [HC-06 BlueTooth Module][6]
7. [MPU-6050 6DOF IMU][7]
8. [Self designed 3D-printed robot structure (on thingiverse)][8]

## »Pin Connection
Refer to [motor.h](https://github.com/xeonqq/balance_robot/blob/kalman/motor.h). Leonardo Serial1 is used to connect the Bluetooth.

## »Dependency
MPU6050 lib from [i2cdevlib](http://github.com/jrowberg/i2cdevlib.git) and [arduino-pinchangeint](https://code.google.com/p/arduino-pinchangeint/downloads/list). You need to place the libs in the ~/sketchbook/libraries/ folder

## »Software Key Components
1. *Kalman filter*: to fuse the data from accerometer and gyro and output angle. The implementation is referenced from TKJElectronics's [Balanduino](https://github.com/TKJElectronics/KalmanFilter).

2. *PID control*: "velocity" PID which can prevent integral windup (Further theory refer to this [link](http://lorien.ncl.ac.uk/ming/digicont/digimath/dpid1.htm)). Target is to stablize the robot around angle 0. 

3. *Tune PID via Bluetooth*: I downloaded [BlueTerm](https://play.google.com/store/apps/details?id=es.pymasde.blueterm&hl=en) to pair and communicate with arduino. Can Tune the PID without wire.

4. *Motor control*: control use PWM signal, a bit tuning is done to make both motors run at same RPM given same PWM value. And the motor has a minimum PWM to be applied, because under certain minimum PWM value the motor does not move at all.

5. *Python Debug*: python script analysis/serial_plot.py can plot the data from Serial port in real time. But the message has to follow the format *"name\t value\t name\t value\t...\n"*

## »Problem encountered
* Initially, becuase I place the IMU with its x-axis pointing forward. So I was calculating the angle of the robot using x and z axis of the accerometer, along with the angle accumulated from gyro in x-axis, but that's wrong! I should use  gyro in y-axis, because rotating around y axis correspond to leaning forward.

* This motor can not supply enough torque when the robot deviate its angle too much, so the robot can still fall down when angle error is big.

 [1]:https://www.arduino.cc/en/Main/ArduinoBoardLeonardo
 [2]:http://world.taobao.com/item/40496339515.htm?fromSite=main&spm=a1z0d.6639537.1997196601.413.U9SqEj
 [3]:http://world.taobao.com/item/20695931042.htm?fromSite=main&spm=a1z0d.6639537.1997196601.4.U9SqEj
 [4]:http://item.taobao.com/item.htm?spm=a312a.7700846.9.323.7gA2vL&id=35296225045&_u=f3e5nn585ef
 [5]:http://www.hobbyking.com/hobbyking/store/__38109__ZIPPY_Flightmax_2200mAh_3S1P_25C_EU_Warehouse_.html
 [6]:http://item.taobao.com/item.htm?spm=a312a.7700846.9.121.rql2Wm&id=19087365613&_u=f3e5nn58d41
 [7]:https://detail.tmall.com/item.htm?id=18635718636&toSite=main
 [8]:http://www.thingiverse.com/thing:969603

