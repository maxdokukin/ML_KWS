import os


def ensure_dir(directory):
    """
    Ensure that a directory exists. Create it if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_tflite_to_C(
        model_id, model_name, model_architecture, inference_type, output_dir,
        window_stride_ms, window_size_ms, dct_coefficient_count, clip_duration_ms, verbose, tflite_path
):
    """
    Converts a TFLite model to a C source file.

    Args:
        model_id: Model identifier.
        model_name: Model name.
        model_architecture: Model architecture.
        inference_type: Inference type.
        output_dir: Directory to save the output C file.
        window_stride_ms: Frame shift in milliseconds.
        window_size_ms: Frame length in milliseconds.
        dct_coefficient_count: Number of MFCC coefficients.
        clip_duration_ms: Total clip duration in milliseconds.
        verbose: Boolean flag for verbose logging.
        tflite_path: Path to the TFLite model file.
    """

    output_path = os.path.join(output_dir, f'{model_id}_{model_name}_{model_architecture}_{inference_type}.cc')

    if verbose:
        print(f"Converting TFLite model to C array and saving to: {output_path}")

    with open(output_path, 'w') as open_file:
        # Write required libraries
        open_file.write('#include <cstdint>\n')
        open_file.write('#include <cstddef>\n')
        open_file.write('#include <stdio.h>\n')
        open_file.write('#include "BufAttributes.h"\n\n')

        # Write model configuration as a C function
        open_file.write("""
void PrintModelConfig()
{
    struct ConfigParam {
        const char* key;
        const char* value;
    };

    ConfigParam config_params[] = {
""")
        params = {
            "model_id": model_id,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "inference_type": inference_type,
            "window_stride_ms": str(window_stride_ms),
            "window_size_ms": str(window_size_ms),
            "dct_coefficient_count": str(dct_coefficient_count),
            "clip_duration_ms": str(clip_duration_ms),
        }

        for key, value in params.items():
            open_file.write(f'        {{"{key}", "{value}"}},\n')

        open_file.write(f"""
    }};

    int num_params = {len(params)};
    printf("+-----------------------------------------------------------------------+\\n");
    printf("|                          Model Configuration                          |\\n");
    printf("+-----------------------------------------------------------------------+\\n");
    for (int i = 0; i < num_params; ++i)
    {{
        printf("%s: %s\\n", config_params[i].key, config_params[i].value);
    }}
    printf("+-----------------------------------------------------------------------+\\n\\n");
}}
""")

        # Write the model data as a C array
        model_arr_name = "g_kwsModel"
        open_file.write(f"static const uint8_t {model_arr_name}[] ALIGNMENT_ATTRIBUTE = ")
        _write_tflite_data(open_file, tflite_path, verbose)

        recording_win = int(clip_duration_ms / window_stride_ms) - 1

        # Write additional deployment functions
        open_file.write(f"""
const uint8_t * GetModelPointer()
{{
    return {model_arr_name};
}}

size_t GetModelLen()
{{
    return sizeof({model_arr_name});
}}

const uint8_t GetFrameShiftMs()
{{
    return {window_stride_ms};
}}

const uint8_t GetFrameLenMs()
{{
    return {window_size_ms};
}}

const uint8_t GetNumMfccCoeffs()
{{
    return {dct_coefficient_count};
}}

const uint8_t GetRecordingWin()
{{
    return {recording_win};
}}
""")

    if verbose:
        print("Conversion to C array completed successfully.")


def _write_tflite_data(open_file, tflite_path, verbose=False):
    """
    Write TFLite model binary data as a C-style array.

    Args:
        open_file: Open file handle to write into.
        tflite_path: Path to the TFLite model file.
        verbose: Boolean flag for verbose logging.
    """
    read_bytes = _model_hex_bytes(tflite_path)
    line = ' {\n\t'
    i = 1
    while True:
        try:
            el = next(read_bytes)
            line += el + ', '
            if i % 20 == 0:
                line += '\n\t'
                open_file.write(line)
                line = ''
            i += 1
        except StopIteration:
            line = line[:-2] + '};\n'
            open_file.write(line)
            break

    if verbose:
        print(f"TFLite data written to file from: {tflite_path}")


def _model_hex_bytes(tflite_path):
    """
    Yield hexadecimal representation of each byte in the TFLite model.

    Args:
        tflite_path: Path to the TFLite model file.

    Yields:
        str: Hexadecimal representation of a byte.
    """
    with open(tflite_path, 'rb') as tflite_model:
        byte = tflite_model.read(1)
        while byte:
            yield f'0x{byte.hex()}'
            byte = tflite_model.read(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TFLite model to C source file.")
    parser.add_argument("--model_id", required=True, help="Model identifier.")
    parser.add_argument("--model_name", required=True, help="Model name.")
    parser.add_argument("--model_architecture", required=True, help="Model architecture.")
    parser.add_argument("--inference_type", required=True, help="Inference type.")
    parser.add_argument("--output_dir", required=True, help="Output directory for the C file.")
    parser.add_argument("--window_stride_ms", type=int, required=True, help="Frame shift in milliseconds.")
    parser.add_argument("--window_size_ms", type=int, required=True, help="Frame length in milliseconds.")
    parser.add_argument("--dct_coefficient_count", type=int, required=True, help="Number of MFCC coefficients.")
    parser.add_argument("--clip_duration_ms", type=int, required=True, help="Total clip duration in milliseconds.")
    parser.add_argument("--tflite_path", required=True, help="Path to the TFLite model file.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")

    args = parser.parse_args()

    convert_tflite_to_C(
        args.model_id, args.model_name, args.model_architecture, args.inference_type,
        args.input_dir, args.output_dir, args.window_stride_ms, args.window_size_ms,
        args.dct_coefficient_count, args.clip_duration_ms, args.verbose, args.tflite_path
    )
