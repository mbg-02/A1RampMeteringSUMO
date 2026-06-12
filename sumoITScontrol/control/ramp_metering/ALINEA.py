"""sumoITScontrol: Traffic Controller Collection for SUMO Traffic Simulations [2026]
Authors: Kevin Riehl <kriehl@ethz.ch>
Organisation: ETH Zürich, Institute for Transport Planning and Systems (IVT)
"""

import numpy as np


class ALINEA:
    def __init__(self, params, ramp_meter):
        self.params = params
        self.ramp_meter = ramp_meter
        # init measurement data
        self.measurement_data = {}
        for sensor in self.ramp_meter.mainline_sensors:
            self.measurement_data[sensor] = {}
            self.measurement_data[sensor]["occupancy_l_interval"] = [[0, 0]]
        self.measurement_data["metering_rate"] = [[0, 100.0]]
        self.measurement_data["metering_occupancy"] = [[0, 0.0]]
        self.measurement_data["queue_length_m"] = [[0, 0]]
        self.measurement_data["queue_ratio"] = [[0, 0.0]]
        self.measurement_data["queue_occupancy"] = [[0, 0.0]]
        self.measurement_data["queue_override_active"] = [[0, 0]]
        self.measurement_data["previous_metering_rate"] = 100.0
        self.measurement_data["previous_occupancy"] = 0.0
        self.measurement_data["counter"] = -1
        self.measurement_data["time"] = 0

    def _get_queue_override_occupancy(self):
        queue_override_state = self.ramp_meter._get_queue_override_state(
            "getLastIntervalOccupancy"
        )
        if len(queue_override_state) == 0:
            return np.nan
        return np.nanmax(queue_override_state)

    def _queue_override_active(self, queue_occupancy):
        threshold = self.params.get("queue_override_occupancy_threshold", 30.0)
        return threshold is not None and queue_occupancy > threshold

    def execute_control(self, current_time):
        """
        This function implements the ALINEA ramp metering controller (which is updating the metering rates).
        This function also conducts necessary measurements of the merging area.
        """
        if not self.ramp_meter.metering_active:
            self.ramp_meter._set_ramp_signal_control_logic_inactive()

        queue_override_occupancy = self._get_queue_override_occupancy()
        queue_override_active = self._queue_override_active(queue_override_occupancy)
        if queue_override_active:
            self.measurement_data["previous_metering_rate"] = (
                self.ramp_meter._set_ramp_signal_control_logic_inactive()
            )

        self.measurement_data["counter"] += 1
        if self.measurement_data["counter"] == self.params["measurement_period"]:
            self.measurement_data["counter"] = 0
            # determine ramp states
            mainline_state = self.ramp_meter._get_mainline_state(
                "getLastIntervalOccupancy"
            )
            queue_state = self.ramp_meter._get_queue_state(
                "getLastIntervalMaxJamLengthInMeters"
            )
            current_occupancy = np.nanmean(mainline_state)
            current_queue_length = np.nansum(queue_state)
            queue_storage_length_m = self.ramp_meter.get_queue_storage_length_m()
            current_queue_ratio = (
                current_queue_length / queue_storage_length_m
                if queue_storage_length_m > 0
                else np.nan
            )
            activation_threshold = self.params.get(
                "activation_threshold",
                self.params["target_occupancy"],
            )
            deactivation_threshold = self.params.get(
                "deactivation_threshold",
                activation_threshold,
            )
            # alinea law
            alinea_metering_rate = (
                self.measurement_data["previous_metering_rate"]
                + self.params["K_P"]
                * (self.params["target_occupancy"] - current_occupancy)
                + self.params["K_I"]
                * (current_occupancy - self.measurement_data["previous_occupancy"])
            )
            alinea_metering_rate = min(
                max(alinea_metering_rate, self.params["min_rate"]),
                self.params["max_rate"],
            )
            # record
            for idx, sensor in enumerate(self.ramp_meter.mainline_sensors):
                self.measurement_data[sensor]["occupancy_l_interval"].append(
                    [current_time, mainline_state[idx]]
                )
            applied_metering_rate = (
                self.ramp_meter._update_ramp_signal_control_logic_alinea_with_activation(
                    metering_rate=alinea_metering_rate / 100,
                    cycle_duration=self.params["cycle_duration"],
                    current_occupancy=current_occupancy,
                    activation_threshold=activation_threshold,
                    deactivation_threshold=deactivation_threshold,
                    queue_override_active=queue_override_active,
                )
            )
            self.measurement_data["metering_rate"].append(
                [current_time, applied_metering_rate]
            )
            self.measurement_data["metering_occupancy"].append(
                [current_time, current_occupancy]
            )
            self.measurement_data["previous_occupancy"] = current_occupancy
            self.measurement_data["previous_metering_rate"] = applied_metering_rate
            self.measurement_data["queue_length_m"].append(
                [current_time, current_queue_length]
            )
            self.measurement_data["queue_ratio"].append(
                [current_time, current_queue_ratio]
            )
            self.measurement_data["queue_occupancy"].append(
                [current_time, queue_override_occupancy]
            )
            self.measurement_data["queue_override_active"].append(
                [current_time, int(queue_override_active)]
            )
