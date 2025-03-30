/**
 * CUDA-AI-Nintendo_NES_Agent - Build Script
 * 
 * This build script compiles CUDA code and links it with the Rust project.
 * It uses the cc crate to compile CUDA files and generate bindings.
 * 
 * The script detects the CUDA installation and configures the build accordingly.
 * 
 * Author: doublegate
 * License: MIT
 */

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=include");

    // Detect CUDA installation
    let cuda_path = detect_cuda_path()
        .expect("Failed to detect CUDA installation. Make sure CUDA toolkit is installed.");
    
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    // Compile CUDA files
    compile_cuda_files(&cuda_path);

    // Generate bindings for CUDA interface
    generate_bindings();
}

fn detect_cuda_path() -> Option<String> {
    // Try to get CUDA path from environment variable
    if let Ok(path) = env::var("CUDA_PATH") {
        return Some(path);
    }

    // Default paths for different platforms
    let default_paths = if cfg!(target_os = "windows") {
        vec![
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2",
        ]
    } else if cfg!(target_os = "linux") {
        vec![
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
            "/usr/local/cuda-11.6",
            "/usr/local/cuda-11.5",
            "/usr/local/cuda-11.4",
            "/usr/local/cuda-11.3",
            "/usr/local/cuda-11.2",
            "/usr/local/cuda-11.1",
            "/usr/local/cuda-11.0",
            "/usr/local/cuda-10.2",
            "/usr/local/cuda",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "/Developer/NVIDIA/CUDA-12.0",
            "/Developer/NVIDIA/CUDA-11.8",
            "/Developer/NVIDIA/CUDA-11.7",
            "/Developer/NVIDIA/CUDA-11.6",
            "/Developer/NVIDIA/CUDA-11.5",
            "/Developer/NVIDIA/CUDA-11.4",
            "/Developer/NVIDIA/CUDA-11.3",
            "/Developer/NVIDIA/CUDA-11.2",
            "/Developer/NVIDIA/CUDA-11.1",
            "/Developer/NVIDIA/CUDA-11.0",
            "/Developer/NVIDIA/CUDA-10.2",
        ]
    } else {
        vec![]
    };

    // Check if any of the default paths exist
    for path in default_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    // Try to get CUDA path using nvcc
    let output = Command::new("which")
        .arg("nvcc")
        .output()
        .ok()
        .and_then(|output| {
            let path = String::from_utf8_lossy(&output.stdout);
            let path = path.trim();
            if !path.is_empty() {
                let mut path_buf = PathBuf::from(path);
                path_buf.pop(); // Remove nvcc binary name
                path_buf.pop(); // Move up from bin directory
                Some(path_buf.to_string_lossy().to_string())
            } else {
                None
            }
        });

    output
}

fn compile_cuda_files(cuda_path: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_files = [
        "cuda/neural_network.cu",
        "cuda/reinforcement.cu",
    ];

    let nvcc_path = if cfg!(target_os = "windows") {
        format!("{}/bin/nvcc.exe", cuda_path)
    } else {
        format!("{}/bin/nvcc", cuda_path)
    };

    for cuda_file in &cuda_files {
        let cuda_file_path = PathBuf::from(cuda_file);
        let file_stem = cuda_file_path.file_stem().unwrap().to_string_lossy();
        let output_file = format!("{}/{}.o", out_dir, file_stem);

        // Build the compile command
        let compute_capabilities = [
            "compute_50", "sm_50",  // Maxwell
            "compute_60", "sm_60",  // Pascal
            "compute_70", "sm_70",  // Volta
            "compute_75", "sm_75",  // Turing
            "compute_80", "sm_80",  // Ampere
            "compute_86", "sm_86",  // Ampere
            "compute_89", "sm_89",  // Ada Lovelace
            "compute_90", "sm_90",  // Hopper
        ];

        let mut gencode_flags = Vec::new();
        for i in (0..compute_capabilities.len()).step_by(2) {
            gencode_flags.push(format!(
                "--generate-code=arch={},code={}",
                compute_capabilities[i], compute_capabilities[i + 1]
            ));
        }

        let include_path = format!("-I{}/include", cuda_path);
        let include_path_project = "-Iinclude";

        let status = Command::new(&nvcc_path)
            .args(&[
                "-O3",
                "--use_fast_math",
                "-std=c++14",
                "-Xcompiler",
                "/MD",
                "--shared",
                "--compiler-options",
                "-fPIC",
                &include_path,
                include_path_project,
                "-c",
                cuda_file,
                "-o",
                &output_file,
            ])
            .args(&gencode_flags)
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to execute nvcc: {}", e);
            });

        if !status.success() {
            panic!("Failed to compile CUDA file: {}", cuda_file);
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static={}", file_stem);
    }

    // Build a C++ wrapper to link all CUDA object files
    let wrapper_file = format!("{}/cuda_wrapper.cpp", out_dir);
    let mut wrapper_content = String::new();
    wrapper_content.push_str("#include <cuda_runtime.h>\n");
    wrapper_content.push_str("extern \"C\" {\n");
    wrapper_content.push_str("    cudaError_t cuda_wrapper_init() { return cudaSuccess; }\n");
    wrapper_content.push_str("}\n");

    std::fs::write(&wrapper_file, wrapper_content).unwrap();

    cc::Build::new()
        .cpp(true)
        .file(&wrapper_file)
        .include(format!("{}/include", cuda_path))
        .include("include")
        .compile("cuda_wrapper");

    // Link the compiled object files
    for cuda_file in &cuda_files {
        let cuda_file_path = PathBuf::from(cuda_file);
        let file_stem = cuda_file_path.file_stem().unwrap().to_string_lossy();
        let output_file = format!("{}/{}.o", out_dir, file_stem);

        cc::Build::new()
            .object(&output_file)
            .compile(&format!("{}", file_stem));
    }
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("include/cuda_interface.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
