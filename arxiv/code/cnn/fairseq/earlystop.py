#!/usr/bin/env python3

lines = [float(line.split()[-1]) for line in open("model.log") if "valid ppl" in line]

num_not_improved = 0
for last_epoch, (last_value, curr_value) in enumerate(zip(lines, lines[1:])):
    epoch = last_epoch + 1
    if curr_value >= last_value:
        num_not_improved += 1
    else:
        # improved
        num_not_improved = 0
    if num_not_improved >= 2:
        with open("early_stop.txt", "w") as out:
            print(epoch, file=out)
        print("Not improved for 2 epochs at checkpoint ", epoch)
        break


