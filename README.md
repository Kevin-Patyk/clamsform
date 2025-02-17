# Clamsform 🐚

* Owner: Kevin Patyk
* Contact: kvn.ptk@gmail.com

This is a personal project which aims to create a wrapper for `polars` to provide preprocessing capabilities such as feature scaling and transformations in Rust🦀. The educational goals are:

* Advance in Rust🦀 by exploring advanced features and performance optimization.
* Expand knowledge of software engineering principles and best practices.
* Develop a package from scratch to learn package development and publishing.
* Gain experience building a back-end in Rust🦀 and translating it to a Python🐍 package.

This repository is in its early stages and will be updated frequently as it evolves.

## Roadmap

* [x] Repository creation and initial files
* [x] Establish initial dependencies and workspace structure
* [x] Implement DataFrame validations
    * [x] Create basic DataFrame validation framework
    * [x] Test basic DataFrame validation framework
* [ ] Implement Z-score standardization
    * [x] Create basic Z-score standardization errors
    * [x] Test basic Z-score standardization errors
    * [ ] Benchmark serial vs parallel execution of validation framework
    * [ ] Create full Z-score standardization implementation
    * [ ] Test full Z-score standardization implementation
* [ ] Implement min-max feature scaling
    
## Installation

```bash
# Clone the repository
git@github.com:Kevin-Patyk/clamsform.git
cd clamsform

# Build the project
cargo build

# Run tests
cargo test
```

## Development Setup

1. Install Rust🦀 (stable)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install nightly Rust🦀 (for formatting & benchmarking)

```bash
rustup toolchain install nightly
```

3. Install Clippy (for linting)

```bash
rustup component add clippy rustfmt
```

## Contributing

While this is primarily a personal project, contributions are welcome from anyone interested in teaching, gaining experience with open source, or simply finding the project interesting. I'd love to learn and collaborate with others.

### Committing

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#specification) for commit messages. Each commit message should be structured as follows:

```
<type>: <description>

[optional body]

[optional footer]
```

Types:

* `feat`: New features or significant changes
* `fix`: Bug fixes
* `docs`: Documentation changes
* `chore`: Maintenance tasks
* `refactor`: Code restructuring without functional changes
* `perf`: Performance improvements
* `build`: Changes to build system or dependencies
* `ci`: Changes to CI configuration

Examples:

```
feat: add MySQL connection pooling
fix: resolve memory leak in parser
docs: update README with new API endpoints
refactor: simplify error handling logic
```

### Branching Strategy

This repository follows a trunk-based branching strategy, with short-lived feature branches making direct pull requests into the `main` branch. 

All branches should follow the conventional commits naming pattern:

```
<type>/<description>
```

Where 'type' follows the commit conventions outlined in the 'Committing' section.

Examples:

```
feat/mysql-pooling
fix/memory-leak
docs/api-reference
refactor/error-handling
```

### Contribution Steps

1. Fork the repository.
2. Create your feature branch following conventional commits.
3. Make your changes.
4. Run the development checks:

```bash
# Run test suite
cargo test

# Format your code (using nightly from rust-toolchain.toml)
cargo fmt

# Run clippy lints
cargo clippy -- -D warnings

# Run benchmarks (if applicable)
cargo bench
```
5. Commit your changes following conventional commits. 
6. Push to your branch:

```bash
git push origin <type>/<description>
```
7. Open a pull request to `main`.

## Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run tests with parallel execution
cargo test -- --test-threads=8
```

## Acknowledgements 

* The Rust🦀 community for tools and support.
* The developers of `polars` and any contributors for making a revolutionary form of working with DataFrames.

## License

Clamsform is MIT licensed. See the LICENSE file for details.
