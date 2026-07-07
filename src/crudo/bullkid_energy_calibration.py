import math
import time

from cirque import ad7291
from cirque import bullkidcalintegration
from cirque import servicehubcontrol
from cirque import servicehubutils

ADC_VOLTAGE_DIVIDER = 0.16967509
MinTap = 0
MaxTap = 63


class BullkidEnergyCalibration:
    def __init__(self, con: servicehubutils.FPGAConnection, plugin):
        self.con = con
        self.cal_cirque = bullkidcalintegration.BullkidCalIntegration(con, plugin)
        self.active_board = 1
        boards = self.detect_boards(
            verbose=True
        )  # automatically sets the correct board slot if only one is connected,
        # uses smaller index if multple are connected
        if boards[0]:
            self.active_board = 1
        elif boards[1]:
            self.active_board = 2
        elif boards[2]:
            self.active_board = 3
        self.cal_cirque.set_active_cal_board(self.active_board)

    def detect_boards(self, verbose=False):
        if verbose:
            print("Connected LANTERN boards:")
        boards = self.cal_cirque.detect_boards()
        board2 = bool((boards >> 2) & 1)
        board1 = bool((boards >> 1) & 1)
        board0 = bool(boards & 1)
        board = (not board0, not board1, not board2)
        if verbose:
            for n in range(3):
                if board[n]:
                    print(f"Slot {n+1}")
            if not any(board):
                print("No board detected!")
        return board

    def set_active_cal_board(self, board):
        boards = self.detect_boards()
        if board < 0 or board > 4:
            print("Board value out of range, possible values: (int) 1, 2, 3")
        else:
            if boards[board - 1]:
                self.active_board = board
                self.cal_cirque.set_active_cal_board(board)
            else:
                print("Chosen board is not connected")

    def configure_calibration(self, led_address, frequency, periods, resistor_tap):
        self.cal_cirque.set_led_address(led_address)
        _ = self.cal_cirque.set_frequency(frequency)
        self.cal_cirque.set_periods(periods)
        self.cal_cirque.set_resistor_tap(resistor_tap)

    def start_calibration(self, single_shot=False):
        self.cal_cirque.triggers(single_shot)

    def enter_calibration_mode(self):
        self.cal_cirque.start_calibration_mode()

    def exit_calibration_mode(self):
        self.cal_cirque.stop_calibration_mode()

    def read_voltage(self):
        adc = ad7291.AD7291(self.con, "adc_fmcp")
        if self.cal_cirque.get_adc_alert():
            print("Warning: ADC-Alert is on!")
            vLED = math.nan
        else:
            if self.active_board == 1:
                in_voltage = adc.get_voltage(4)
            elif self.active_board == 2:
                in_voltage = adc.get_voltage(5)
            elif self.active_board == 3:
                in_voltage = adc.get_voltage(7)
            vLED = round(in_voltage * (1 / ADC_VOLTAGE_DIVIDER), 3)
            print(vLED)
        return vLED

    def measure_temperature(self):
        adc = ad7291.AD7291(self.con, "adc_fmcp")
        temp = adc.get_temperature()
        print(f"FMCP Board Temperature = {temp} °C")

    def digital_resistor_test(self):
        print(f"Digital Resistor Sweep from {MinTap} to {MaxTap} and back")
        print("VLED Voltages:")
        for i in range(MinTap, MaxTap + 1, 1):
            self.cal_cirque.set_resistor_tap(i)
            time.sleep(0.02)
            self.read_voltage()
        for i in range(MaxTap, MinTap - 1, -1):
            self.cal_cirque.set_resistor_tap(i)
            time.sleep(0.02)
            self.read_voltage()
