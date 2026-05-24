#include "global.h"
class Tokenizer {
private:
	StdVec<std::string> token_table_;
	bool init_ok_;

public:
	int eot_token;

public:
	Tokenizer() {}
	Tokenizer(const std::filesystem::path &checkpoint_path);
	StdVec<int> encode(const std::string &str);
	std::string decode(const StdVec<int> &);
	std::string decode(int code);
	operator bool() const{return init_ok_;};
};
