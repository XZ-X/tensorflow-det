#ifndef TENSORFLOW_CORE_UTIL_OP_LOGGER_H_
#define TENSORFLOW_CORE_UTIL_OP_LOGGER_H_

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/graph/graph.h"

using namespace tensorflow;

class OpLogger {
  public:

    static OpLogger& getInstance();

    uint64 GetAndAddIterationByName(const std::string& name);
    
    uint64 GetIterationByName(const std::string& name);
    
    void ResetIteration(const std::string& name);
    
    void ResetIteration(const std::string& name, uint64 counter);
    
    void AddIterator(const std::string& name, void* it);
    
    void RemoveIterator(const std::string& name);
   
    void SaveOrRestoreIterations(const std::string& name);

    void SaveOrRestoreIterators(const std::string& name, void* ctx, bool is_all=false); 

  private:
    OpLogger() {}
    
    bool RestoreIterator(const std::string& name, void* ctx);
    
    void SaveIterators();

    std::ofstream default_file;
    std::ofstream graph_file;
    std::unordered_map<std::string, uint64> counter_; 
    mutex mu_;
    std::unordered_map<std::string, void*> iters_; 
    std::unordered_map<std::string, std::string> del_iters_; 
    mutex it_mu_;
    std::unordered_map<std::string, void*> saved_iters_; 
    mutex sit_mu_;
    bool in_library = false;
};

#endif  // TENSORFLOW_CORE_UTIL_TRACE_UTIL_H_
