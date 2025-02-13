# Clamsform🐚

* Owner: Kevin Patyk
* Contact: kvn.ptk@gmail.com

This is a personal project which aims to create a wrapper for `polars` to provide preprocessing capabilities such as feature scaling and transformations in Rust🦀. The educational goals are:

* Become more advanced in Rust programming.
* Expand upon software engineering principles and best practices.
* Learn more about the mathematics behind machine learning algorithms.
* Learn how to code more machine learning algorithms from scratch.
* Get experience in developing a back-end in Rust🦀 and translating it to a package in Python🐍.

This repository is still in its starting stages, but will continue to grow, thus it will be updated frequently and is subject to change. 

# Installation

```bash
# Clone the repository
git@github.com:Kevin-Patyk/clamsform.git
cd clamsform

# Build the project
cargo build

# Run tests
cargo test
```

## Branching Strategy

This repository follows a trunk-based branching strategy, with short-lived feature branches making direct pull requests into the `main` branch. This ensures:

* Rapid integration of new features.
* Reduced merge conflicts.
* Continuous delivery of improvements.
* Simple project maintenance.

## Development Setup

1. Install Rust🦀 (stable)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install nightly Rust🦀 (for formatting)
```bash
rustup toolchain install nightly
rustup component add rustfmt --toolchain nightly
```

## Contributing

Although this is more of a personal project, contributions are always welcome for those wanting to teach others, gain experience in working on open source projects, or if you find it interesting! It would be great to learn and work with others.

### Contribution Guidelines

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Make your changes.
4. Run the test suite:
```bash
cargo test
```
5. Format your code using nightly Rust🦀:
```bash
cargo +nightly fmt
```
6. Lint your code using `clippy`:
```bash
cargo clippy -- -D warnings
```
7. Commit your changes (`git commit -m 'Add Some Amazing Feature`).
8. Push to the branch (`git push origin feature/amazing_feature`).
9. Open a pull request.

## Running Tests
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

## Acknowledgements 

## Acknowledgements 

* Scikit-learn for inspiration and providing a great reference implementation.
* The Rust🦀 community for tools and support.
* The developers of `polars` and any contributors for making a revolutionary form of working with DataFrames.

## License

Clamsform is MIT licensed. See the LICENSE file for details.
The MIT License is a permissive license that allows for reuse with few restrictions. It permits users to:

* Use the code commercially.
* Modify the code.
* Distribute the code.
* Use the code privately.
* Sublicense the code.

The only requirement is that the license and copyright notice must be included with the code.
