#pragma once
#include <string>

namespace tf {

class Module;

void save_checkpoint(const Module &model, const std::string &path);
void load_checkpoint(Module &model, const std::string &path);

}  
