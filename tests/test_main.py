from coremlprofiler import CoreMLProfiler, ComputeDevice, ComputeUnitSetting
from huggingface_hub import snapshot_download


def test_integration_with_real_model():
    repo_path = snapshot_download(repo_id="FL33TW00D-HF/test-st", local_dir="./")
    model_path = repo_path + "/sentence_transformer_all-MiniLM-L6-v2.mlpackage"

    profiler = CoreMLProfiler(model_path)

    summary = profiler.device_usage_summary()
    assert summary[ComputeDevice.CPU] > 0
    # Some models might not use ANE, so we shouldn't assert it
    print(profiler.device_usage_summary_chart())

def test_compute_unit_settings():
    repo_path = snapshot_download(repo_id="FL33TW00D-HF/test-st", local_dir="./")
    model_path = repo_path + "/sentence_transformer_all-MiniLM-L6-v2.mlpackage"

    # Test CPU only mode
    cpu_profiler = CoreMLProfiler(model_path, compute_units=ComputeUnitSetting.CPU_ONLY)
    cpu_summary = cpu_profiler.device_usage_summary()
    assert cpu_summary[ComputeDevice.CPU] > 0    
    assert cpu_summary[ComputeDevice.GPU] == 0  # GPU should definitely be unused

    # Test CPU and Neural Engine mode
    ne_profiler = CoreMLProfiler(model_path, compute_units=ComputeUnitSetting.CPU_AND_NE)
    ne_summary = ne_profiler.device_usage_summary()
    assert ne_summary[ComputeDevice.CPU] > 0
    assert ne_summary[ComputeDevice.GPU] == 0  # GPU should definitely be unused
