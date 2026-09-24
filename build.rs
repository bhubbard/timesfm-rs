use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        println!("cargo:rustc-link-search=native=/usr/lib/swift");

        if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
            if let Ok(developer_dir) = String::from_utf8(output.stdout) {
                let developer_dir = developer_dir.trim();
                let swift_55_dir = format!(
                    "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift-5.5/macosx",
                    developer_dir
                );
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", swift_55_dir);
                println!("cargo:rustc-link-search=native={}", swift_55_dir);

                let swift_dir = format!(
                    "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
                    developer_dir
                );
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", swift_dir);
                println!("cargo:rustc-link-search=native={}", swift_dir);
            }
        }
    }
}
