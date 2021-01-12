import glob
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


def read_tb(event_folder, tag):
    files = glob.glob(f"{event_folder}/*")
    if not 0 < len(files) < 2:
        print(f"Event folder doesn't contain one file. {event_folder}")
        return None
    event_file = files[0]
    print(f"Load {event_file}")
    loader = EventFileLoader(event_file)
    step = -1
    steps = []
    wtimes = []
    values = []
    for event in loader.Load():
        if len(event.summary.value)==0:
            continue
        summary = event.summary.value[0]
        if summary.tag!=tag:
            continue
        step = event.step
        wtime   = event.wall_time
        value = summary.tensor.float_val[0]
        # print(f"step {step}: wtime {wtime}, value {value}")
        steps.append(step)
        wtimes.append(wtime)
        values.append(value)
    return {"steps": steps, "wtimes": wtimes, "values": values}
