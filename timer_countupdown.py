import tkinter as tk
import subprocess
import os
import csv
import pandas as pd
# os.chdir()
# print(__file__)
# print(os.getcwd())
# print(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
file = 'totaly_homeworks_time.csv'
file2 = 'WeeklyGoals.csv'
# font = 'Monaco'
hour_const = 0
minite_const = 0
second_const = 3
hour = hour_const
minite = minite_const
second = second_const
total_sec = hour * 3600 + minite * 60 + second



def start(event):
    global state, on_off, total_sec, total_time,font, on_off_byouyomi,bg_byouyomi
    if state == True:
        state = False
        print("False")
        subprocess.Popen(['caffeinate', '-i']).terminate()
        # subprocess.Popen(['caffeinate', '-i', '-d']).terminate()
    else:
        state = True
        print("True")
        subprocess.Popen(['caffeinate', '-i'])
        # subprocess.Popen(['caffeinate', '-i', '-d'])

        if on_off == True:
            x = total_sec
            # print(x)
            if os.path.exists(file) == True:
                ddf = pd.read_csv(file, header=0, engine='python')
                total_time = ddf.iloc[0,0]
                renewed_time = total_time + x
                # print(total_time + x)
                total_minites = renewed_time // 60
                total_hours = renewed_time // 3600
                total_days = renewed_time // (3600 * 24)
                with open(file,mode='w',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['second', 'minite', 'hour', 'day'])
                    writer.writerow([renewed_time, total_minites, total_hours, total_days])
                ddf2 = pd.read_csv(file2, header=0, engine='python')
                total_time2 = ddf2.iloc[0,1] * 3600
                renewed_time2 = total_time2 + x
                # print(total_time2 + x)
                # total_minites = renewed_time // 60
                total_hours2 = renewed_time2 // 3600
                with open(file2,mode='w',newline='') as f2:
                    writer = csv.writer(f2)
                    writer.writerow(['weekly goals', 'studytime'])
                    writer.writerow([14, total_hours2])

            else:
                with open(file,mode='w',newline='') as f:
                    total_minites = x // 60
                    total_hours = x // 3600
                    total_days = x // (3600 * 24)
                    writer = csv.writer(f)
                    writer.writerow(['second', 'minite', 'hour', 'day'])
                    writer.writerow([x, total_minites, total_hours, total_days])

            on_off = False
            on_off_byouyomi = False
            # bg_byouyomi = '#000000'
            on_offText.configure(font=font,text='OVERWRITE:{}'.format(str_on_off))
            on_offByouyomi.configure(font=font,text='秒読み:{}'.format(str_on_off_byouyomi))
            # on_offByouyomi['bg'] = '#808080'
            ddf = pd.read_csv(file, header=0, engine='python')
            total_time = ddf.iloc[0,0]
            secText.configure(font=font,text='TOTAL: {:>9,.1f} hours'.format(total_time/(60*60)))
        else:
            # on_off = True
            pass

def update_timeText():
    if (state):
        global timer, total_sec,updown,on_off_byouyomi,num_byouyomi
        if total_sec > 0:
            total_sec += updown
            timer[0] = total_sec // 3600
            timer[1] = (total_sec - 3600 * timer[0]) // 60
            timer[2] = total_sec % 60

        if total_sec <= 0:
            clear_command = 'cls'
            subprocess.run([clear_command], shell=True)
            print('\a')

        if on_off_byouyomi == True:
            if total_sec%num_byouyomi == 1 or total_sec%num_byouyomi == 2 or total_sec%num_byouyomi == 3 :
                clear_command = 'cls'
                subprocess.run([clear_command], shell=True)
                print('\a')

        timeString = pattern.format(timer[0], timer[1], timer[2])
        timeText.configure(text=timeString)

        on_offText.configure(font=font,text='OVERWRITE:{}'.format(str_on_off))
    root.after(1000, update_timeText)


def reset():
    global timer, total_sec, state, on_off, total_time,font,updown,on_off_byouyomi,bg_byouyomi,num_byouyomi
    # font = 'Monaco'
    if state == True:
        state = False
        print("False")
        subprocess.Popen(['caffeinate', '-i']).terminate()
        # subprocess.Popen(['caffeinate', '-i', '-d']).terminate()
    on_off = True
    on_off_byouyomi = False
    entry.delete(0,tk.END)
    entry.insert(tk.END,num_byouyomi)
    bg_byouyomi = '#000000'
    updown = -1
    cs = 'COUNTDOWN'
    countstateButton['text'] = '{}'.format(cs)
    hour = hour_const
    minite = minite_const
    second = second_const
    total_sec = hour * 3600 + minite * 60 + second
    timer = [hour, minite, second, 0]
    timeText.configure(text='{:0=2}:{:0=2}:{:0=2}'.format(hour, minite, second))
    on_offText.configure(font=font,text='OVERWRITE:{}'.format(str_on_off))
    on_offByouyomi.configure(font=font,text='秒読み:{}'.format(str_on_off_byouyomi))
    # on_offByouyomi['bg'] = '#808080'
    ddf = pd.read_csv(file, header=0, engine='python')
    total_time = ddf.iloc[0,0]
    secText.configure(font=font,text='TOTAL: {:>9,.1f} hours'.format(total_time/(60*60)))

# To exist our program
def exist():
    root.destroy()

def switch():
    global on_off,font
    if on_off == True:
        on_off = False
        str_on_off = "OFF"
    else:
        on_off = True
        str_on_off = "ON "
    on_offText.configure(font=font,text='OVERWRITE:{}'.format(str_on_off))

def switch_byouyomi():
    global on_off_byouyomi,font,bg_byouyomi,on_offByouyomi
    if on_off_byouyomi == True:
        on_off_byouyomi = False
        bg_byouyomi = '#000000'
        str_on_off_byouyomi = "OFF"
    else:
        on_off_byouyomi = True
        bg_byouyomi = '#808080'
        str_on_off_byouyomi = "ON "
        num_byouyomi = entry.get()
        # print(num_byouyomi)
    # on_offByouyomi['bg'] = bg_byouyomi
    on_offByouyomi.configure(font=font,text='秒読み:{}'.format(str_on_off_byouyomi))


# def switchupdown():
#     global on_off,updown
#     if updown == 1:
#         updown = -1
#         str_updown = "DOWN"
#     else:
#         updown = 1
#         str_updown = "UP  "
#     on_offText.configure(font=font,text='OVERWRITE:{}'.format(str_on_off))

def switchupdown():
    global updown
    print("T")
    if updown == 1:
        updown = -1
        cs = 'COUNTDOWN'
    else:
        updown = 1
        cs = 'COUNTUP'
    countstateButton['text'] = '{}'.format(cs)
    root.wm_title('{} TIMER'.format(cs))
    # print(updown)

# def countstate():
#     global count_state, cs
#     if count_state == True:
#         count_state = False
#         cs = 'COUNTUP'
#     else:
#         count_state = True
#         cs = 'COUNTDOWN'
#     # countstateButton['text'] = '{}'.format(cs)
#     # print(count_state)

def add(min):
    global total_sec
    total_sec += min * 60
    timer[0] = total_sec // 3600
    timer[1] = (total_sec - 3600 * timer[0]) // 60
    timer[2] = total_sec % 60
    timeString = pattern.format(timer[0], timer[1], timer[2])
    timeText.configure(text=timeString)

def sub(min):
    global total_sec
    if total_sec >= min * 60:
        total_sec -= min * 60
    timer[0] = total_sec // 3600
    timer[1] = (total_sec - 3600 * timer[0]) // 60
    timer[2] = total_sec % 60
    timeString = pattern.format(timer[0], timer[1], timer[2])
    timeText.configure(text=timeString)

def minites(min):
    global total_sec, active_button
    total_sec = min * 60 + 3
    timer[0] = total_sec // 3600
    timer[1] = (total_sec - 3600 * timer[0]) // 60
    timer[2] = total_sec % 60
    timeString = pattern.format(timer[0], timer[1], timer[2])
    timeText.configure(text=timeString)

    active_button = button_dict[min]
    if active_button.cget('fg') == 'blue':
        active_button.config(fg='black')
    else:
        active_button.config(fg='blue')

def minbutton(min,row,sticky,padx):
    tk.Button(root,font=font,text='{}min'.format(min), command=lambda:minites(min)).grid(row=row, sticky=sticky, padx=padx)
def addbutton(min,row,sticky,padx):
    tk.Button(root,font=font,text='+{}'.format(min), command=lambda:add(min)).grid(row=row, sticky=sticky, padx=padx)

def subbutton(min,row,sticky,padx):
    tk.Button(root,font=font,text='-{}'.format(min), command=lambda:sub(min)).grid(row=row, sticky=sticky, padx=padx)

def mminbutton(min, row, column):
    button = tk.Button(root, font=font, text='{}min'.format(min), command=lambda: minites(min))
    button.grid(row=row, column=column)
    button_dict[min] = button  # ボタンを辞書に追加

def maddbutton(min,row,column):
    tk.Button(root,font=font,text='+{}'.format(min), command=lambda:add(min)).grid(row=row, column=column)

def msubbutton(min,row,column):
    tk.Button(root,font=font,text='-{}'.format(min), command=lambda:sub(min)).grid(row=row,column=column)



state = False
on_off = True
on_off_byouyomi = False
num_byouyomi = 20

bg_byouyomi = '#000000'
count_state = True
updown = -1
cs = 'COUNTDOWN'
str_on_off = "ON "
str_on_off_byouyomi = "OFF"
font = 'Monaco'
ddf = pd.read_csv(file, header=0, engine='python')
total_time = ddf.iloc[0,0]

# print(os.getcwd())
width = int(470)
height = int(160)
if os.getcwd() == r"C:\Users\tokun\iCloudDrive\dist":
    width = int(360)
    height = int(190)
    font = ('Consolas',9)
    # font = ('Helvetica',9)
root = tk.Tk()
root.wm_title('{} TIMER'.format(cs))
root.geometry('-0+0')
root.geometry('%sx%s'% (width,height))

root.bind("<space>",start)

# Our time structure [min, sec, centsec]
timer = [hour, minite, second, 0]
# The format is padding all the
pattern = '{0:02d}:{1:02d}:{2:02d}'

active_button = None
button_dict = {}  # ボタンを格納する辞書

# Create a timeText Label (a text box)
timeText = tk.Label(root, text='{:0=2}:{:0=2}:{:0=2}'.format(hour, minite, second), font=("Helvetica", 50))
on_offText = tk.Button(root,font=font,text='OVERWRITE:{}'.format(str_on_off), command=switch)
on_offByouyomi = tk.Button(root,font=font,text='秒読み:{}'.format(str_on_off_byouyomi), command=switch_byouyomi)
secText = tk.Label(root,font=font,text='TOTAL: {:>9,.1f} hours'.format(total_time/(60*60)))
startButton = tk.Button(root,font=font,text='Start/Stop', command=lambda:start(state))
resetButton = tk.Button(root,font=font,text='Reset', command=reset)
# onoffButton = tk.Button(root,font=font,text='on/off', command=switch)
countstateButton = tk.Button(root,font=font,text='{}'.format(cs), command=switchupdown)
entry = tk.Entry(root,width=5)
entry.insert(tk.END,num_byouyomi)
entry.place(x=width*0.83, y=height*0.1)

# quitButton = tk.Button(root, text='Quit', command=exist)
# quitButton.grid(row=3)


# if os.getcwd() == "/Users/tokuni/Library/Mobile Documents/com~apple~CloudDocs/dist":
timeText.grid(row=0, column=1,columnspan=3)
on_offText.grid(row=4, column=3)
on_offByouyomi.grid(row=4, column=4)
secText.grid(row=4, column=0,columnspan=2)
startButton.grid(row=1, column=2)
resetButton.grid(row=2, column=2)
# onoffButton.grid(row=3, column=2)
countstateButton.grid(row=4, column=2)

mminbutton(5,1,0)
mminbutton(10,1,1)
mminbutton(25,1,3)
mminbutton(30,1,4)
mminbutton(50,2,0)
mminbutton(60,2,1)
mminbutton(75,2,3)
mminbutton(120,2,4)
maddbutton(1,3,0)
maddbutton(5,3,1)
msubbutton(1,3,3)
msubbutton(5,3,4)
# maddbutton(10,3,3)
# msubbutton(10,3,tk.W,10)

    # timeText.grid(row=0)
    # on_offText.grid(row=4, sticky=tk.E, padx =5)
    # secText.grid(row=4, sticky=tk.W, padx =5)
    # startButton.grid(row=1)
    # resetButton.grid(row=2)
    # onoffButton.grid(row=3)
    # minbutton(1,1,tk.W,10)
    # minbutton(10,1,tk.W,55)
    # minbutton(25,1,tk.E,55)
    # minbutton(35,1,tk.E,5)
    # minbutton(45,2,tk.W,5)
    # minbutton(50,2,tk.W,55)
    # minbutton(75,2,tk.E,55)
    # minbutton(90,2,tk.E,5)
    # addbutton(1,3,tk.E,80)
    # addbutton(5,3,tk.E,50)
    # addbutton(10,3,tk.E,10)
    # subbutton(1,3,tk.W,80)
    # subbutton(5,3,tk.W,50)
    # subbutton(10,3,tk.W,10)

update_timeText()

root.mainloop()
