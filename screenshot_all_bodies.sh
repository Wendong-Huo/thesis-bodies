#!/bin/sh
set -x

for id in $(seq 0 99);
do
    echo $id
    python show.py -n $id > /dev/null &
    sleep 1
    shot -c 550x680+2530+150 -f screenshots/body_$id.png -t screenshots/_tmp.png
    xdotool search ExampleBrowser | xargs xdotool windowkill
done
