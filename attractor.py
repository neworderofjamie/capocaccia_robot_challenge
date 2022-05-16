import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import (GeNNModel, create_custom_current_source_class, create_custom_neuron_class, create_custom_init_var_snippet_class,
                               create_dpf_class, init_connectivity, init_var)

from copy import deepcopy

lif_sfa_model = create_custom_neuron_class(
    "lif_sfa",
    param_names=["C", "TauM", "TauSFA", "Vrest", "Vreset", "Vthresh", "Esfa", "Dsfa", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("Gsfa", "scalar"), ("RefracTime", "scalar")],
    derived_params=[("ExpTC", create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("ExpSFA", create_dpf_class(lambda pars, dt: np.exp(-dt / pars[2]))()),
                    ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])())],
                     
    sim_code=
        """
        if ($(RefracTime) <= 0.0) {
            const scalar iSFA = $(Gsfa) * ($(V) - $(Esfa));
            const scalar alpha = (($(Isyn) - iSFA) * $(Rmembrane)) + $(Vrest);
            $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
        }
        else {
            $(RefracTime) -= DT;
        }

        $(Gsfa) *= $(ExpSFA);
        """,
    threshold_condition_code=
        """
        $(RefracTime) <= 0.0 && $(V) >= $(Vthresh)
        """,
    reset_code=
        """
        $(V) = $(Vreset);
        $(RefracTime) = $(TauRefrac);
        $(Gsfa) += $(Dsfa);
        """,
    is_auto_refractory_required=False)
    
exc_weight_init_model = create_custom_init_var_snippet_class(
    "exc_weight_init",
    param_names=["J0", "SigmaConn", "NeuronTheta", "g"],
    var_init_code=
        """
        const scalar preTheta = $(id_pre) * $(NeuronTheta);
        const scalar postTheta = $(id_post) * $(NeuronTheta);
        const scalar thetaDiff = preTheta - postTheta;
        $(value) = $(g) * ($(J0) / (2.50662827463 * $(SigmaConn))) * exp(-(thetaDiff * thetaDiff) / (2 * $(SigmaConn) * $(SigmaConn)));
        """)
        
uniform_motion_current_source_model = create_custom_current_source_class(
    "uniform_motion_current_source",
    param_names=["Lambda", "SigmaConn", "NeuronTheta"],
    var_name_types=[("I", "scalar")],
    extra_global_params=[("ThetaStim", "scalar")],
    injection_code=
        """
        const scalar theta = $(id) * $(NeuronTheta);
        const scalar thetaDiff = theta - $(ThetaStim);
        $(I) = $(Lambda) * (-0.08 + 0.48 * exp(-(thetaDiff * thetaDiff) / (2 * $(SigmaConn) * $(SigmaConn))));
        $(injectCurrent, $(I) + (5.0 * $(gennrand_normal)));
        """)
        
# Neuron parameters
lif_sfa_params = {"TauSFA": 90.0, "Vrest": -70.0, "Vreset": -60.0,
                  "Vthresh": -50.0, "Esfa": -80.0, "Dsfa": 0.004}
exc_params = deepcopy(lif_sfa_params)
exc_params.update({"C": 0.5, "TauM": 20.0, "TauRefrac": 2.0})

inh_params = deepcopy(lif_sfa_params)
inh_params.update({"C": 0.2, "TauM": 10.0, "TauRefrac": 1.0})

# Neuron initialisation
lif_sfa_init = {"V": lif_sfa_params["Vreset"], "Gsfa": 0.0, "RefracTime": 0.0}

NE = 128
NI = 32

SIGMA_CONN_DEG = 30
NEURON_THETA_DEG = 360.0 / NE

# Excitatory synapse params
exc_psm_params = {"tau": 30.0, "E": 0.0}
inh_psm_params = {"tau": 10.0, "E": -70.0}

exc_weight_init_params = {"J0": 0.3, "SigmaConn": SIGMA_CONN_DEG , "NeuronTheta" : NEURON_THETA_DEG, "g": 0.0571}

input_cs_params = {"Lambda": 1.3 ,"SigmaConn": SIGMA_CONN_DEG, "NeuronTheta": NEURON_THETA_DEG}

model = GeNNModel("float", "continuation_attractor")
model.dT = 0.1

exc = model.add_neuron_population("Exc", NE, lif_sfa_model, exc_params, lif_sfa_init)
inh = model.add_neuron_population("Inh", NI, lif_sfa_model, inh_params, lif_sfa_init)

exc.spike_recording_enabled = True

input = model.add_current_source("Input", uniform_motion_current_source_model, exc,
                                 input_cs_params, {"I": 0.0})

model.add_synapse_population("ExcExc", "DENSE_INDIVIDUALG", 0,
                             exc, exc,
                             "StaticPulse", {}, {"g": init_var(exc_weight_init_model, exc_weight_init_params)}, {}, {},
                             "ExpCond", exc_psm_params, {})

model.add_synapse_population("ExcInh", "DENSE_GLOBALG", 0,
                             exc, inh,
                             "StaticPulse", {}, {"g": 0.0175 / NE}, {}, {},
                             "ExpCond", inh_psm_params, {})

model.add_synapse_population("InhInh", "DENSE_GLOBALG", 0,
                             inh, inh,
                             "StaticPulse", {}, {"g": 0.0082 / NI}, {}, {},
                             "ExpCond", inh_psm_params, {});

model.build()
model.load(num_recording_timesteps=100000)

theta_stim_view = input.extra_global_params["ThetaStim"].view
i_view = input.vars["I"].view
theta_stim_view[:] = 0.0
model.timestep = 0
model.time = 0.0
isyn = []
while model.timestep < 100000:
    model.step_time()
    input.pull_var_from_device("I")
    isyn.append(np.copy(i_view))
    
    theta_stim_view[:] = np.fmod(theta_stim_view[:] + 70.0 * 0.1E-3, 360.0)

model.pull_recording_buffers_from_device()

spike_times, spike_ids = exc.spike_recording_data
print(spike_times.shape)

fig, axes = plt.subplots(2)

axes[0].scatter(spike_times, spike_ids, s=1)

isyn = np.vstack(isyn).T
axes[1].imshow(isyn, aspect=100)

print(np.amin(isyn), np.amax(isyn))
plt.show()