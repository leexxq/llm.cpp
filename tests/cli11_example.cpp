#include "CLI/CLI.hpp"

int main(int argc, char **argv) {
	CLI::App app;
	// Add new options/flags here

	bool my_flag1{ false };
	app.add_flag("--f1", my_flag1, "flag 1");
	int my_flag2{ 0 };
	app.add_flag("--f2", my_flag2, "flag 2");

	auto callback = [](int count) { std::cout << "This was called " << count << " times"; };

	app.add_flag_function("-c", callback, "callback function");

	CLI::Option *flag_plain = app.add_flag("--plain,-p", "This is a plain flag");

	CLI11_PARSE(app, argc, argv);

	return 0;
}
