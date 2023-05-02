extern crate built;
extern crate project_root;

fn main() {
    let src = project_root::get_project_root().unwrap();
    let dst = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("built.rs");
    let mut opts = built::Options::default();

    opts.set_dependencies(true);
    // opts.set_compiler(true);
    opts.set_env(true);
    built::write_built_file_with_opts(&opts, std::path::Path::new(&src), &dst)
        .expect("Failed to acquire build-time information");
    // built::write_built_file()
    //     .expect("Failed to acquire build-time information");
}
