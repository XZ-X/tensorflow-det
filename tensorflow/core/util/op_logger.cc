
#include "tensorflow/core/util/op_logger.h"

#include <sys/types.h>
#include <unistd.h>
#include <string>

#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/graph/edgeset.h"

__attribute__((constructor)) void initializer() {
  // Register the exit function
  OpLogger::getInstance();
}

OpLogger& OpLogger::getInstance() {
  static char buf[sizeof(OpLogger)];
  static OpLogger* theOneTrueObject = new (buf) OpLogger();
  return *theOneTrueObject;
}

void OpLogger::SaveOrRestoreIterations(const std::string& name) {
  const char* cmd = getenv("TF_CHECKPOINT_ITREATION_CMD");
  if(cmd!=NULL){
    const char* path = getenv("TF_CHECKPOINT_ITERATION_SAVEPATH");
    std::string file_path = std::string(path)+name;

    int cmd_type = atoi(cmd);
    switch (cmd_type) {
      case 0:
        break;
      case 1:
        {
          fprintf(stderr, "Start to Restore Iteration counter %s\n", file_path.c_str());
          std::ifstream ifile;
          ifile.open(file_path.c_str());
          if (!ifile.is_open()){
            fprintf(stderr, "Fail to open %s\n", file_path.c_str());
          } else {
            uint64 counter;
            ifile >> counter;
            counter_[name] = counter;
            ifile.close();
          }
          setenv("TF_CHECKPOINT_ITREATION_CMD", "0", 1);
        }
        break;
      case 2:
        {
          fprintf(stderr, "Start to save Iteration counter %s\n", file_path.c_str());
          std::ofstream ofile;
          ofile.open(file_path.c_str(), std::ofstream::trunc);
          ofile << counter_[name];
          ofile.close();
          setenv("TF_CHECKPOINT_ITREATION_CMD", "0", 1);
        }
        break;
      default:
        fprintf(stderr, "Iteration cmd %d can not be handled!!!", cmd_type);
    }
  }
}

uint64 OpLogger::GetAndAddIterationByName(const std::string& name) {
  if(counter_.find(name)==counter_.end())
    counter_[name] = 0;
  return counter_[name]++;
}

uint64 OpLogger::GetIterationByName(const std::string& name) {
  return counter_[name];
}
    
void OpLogger::ResetIteration(const std::string& name) {
  if(in_library) return;

  if(counter_.find(name)==counter_.end())
    counter_[name] = 0;
  
  uint64 upper_bits = 4294967296; //2^32
  counter_[name] &= (~(upper_bits-1));
  counter_[name] += upper_bits;
  fprintf(stderr, "name [%s], counter %lx\n", name.c_str(), counter_[name]);
}

void OpLogger::ResetIteration(const std::string& name, uint64 counter) {
  counter_[name] = counter;
}

void OpLogger::AddIterator(const std::string& name, void* it) {
  if(in_library) return;

  /*
  size_t sub_iter = name.find("[");
  size_t sub_iter1 = name.find("]");
  // this is a sub-iterator
  if(sub_iter!=std::string::npos && sub_iter1!=std::string::npos) {
    return;
  }
  */
  mutex_lock lock(it_mu_);
  if(iters_.find(name)!=iters_.end()) {
    fprintf(stderr, "WARNING: duplicated key %s\n", name.c_str());
    return;
  }
  /*
  for(const auto& iter : iters_) {
    //name: iter.first;
    size_t pos = name.find_last_of("::");
    if(name.substr(0, pos)==iter.first) {
      void* old_iter = iter.second;
      iters_.erase(iter.first);
      // keep the first level iterator
      iters_[name] = old_iter;
      return;
    }
  }
  */
  // new iterator
  iters_[name] = it;
  fprintf(stderr, "new iterator %s, %p\n", name.c_str(), it);
}

void OpLogger::RemoveIterator(const std::string& name) {
  if(in_library) return;

  mutex_lock lock(it_mu_);
  if(iters_.find(name)!=iters_.end()) {
    /*
       size_t pos = name.find_last_of("::");
       void* old_iter = iters_[name];
       iters_.erase(name);  
       if(std::count(name.begin(), name.end(), ':') > 2) {
       iters_[name.substr(0, pos)] = old_iter;
       }
       */

    VariantTensorData data;
    data.set_type_name(name);
    data::VariantTensorDataWriter writer(&data);
    SerializationContext serialization_ctx({});
    IteratorBase* iterbase = (IteratorBase*)iters_[name]; 
    Status s = iterbase->Save(&serialization_ctx, &writer);
    if(!s.ok()) {
      fprintf(stderr, "WARNING: Save iterator failed!!! %s\n", name.c_str());
    }
    writer.Flush();

    std::string data_str = data.SerializeAsString();
    size_t size = data_str.size();
    std::string file_path = std::string("./temp/") + name;
    std::ofstream ofile;
    ofile.open(file_path.c_str(), std::ofstream::trunc | std::ofstream::binary);
    ofile.write(const_cast<char*>(data_str.data()), size);
    ofile.close();

    del_iters_[name] = file_path; 

    iters_.erase(name);  
    fprintf(stderr, "remove iterator %s\n", name.c_str());
  }
}

void OpLogger::SaveOrRestoreIterators(const std::string& name, void* ctx, bool is_all) {
  if(in_library) return;

  const char* cmd = getenv("TF_CHECKPOINT_ITER_CMD");
  if(cmd!=NULL){
    int cmd_type = atoi(cmd);
    switch (cmd_type) {
      case 0:
        break;
      case 1:
        {
          mutex_lock lock(sit_mu_);
          bool current_state = in_library;
          if(!current_state) {
            in_library = true;
          }

          if(is_all) {
            for(const auto& iter : iters_) {
              RestoreIterator(iter.first, ctx);
            }
            setenv("TF_CHECKPOINT_RNG_CMD", "0", 1);
          } else {
            if(RestoreIterator(name, ctx)) {
              setenv("TF_CHECKPOINT_RNG_CMD", "0", 1);
            } else {
              setenv("TF_CHECKPOINT_RNG_CMD", "1", 1);
            }
          }
          
          if(!current_state) {
            in_library = false;
          }
        }
        break;
      case 2:
        SaveIterators(); 
        break;
      default:
        fprintf(stderr, "Cmd %d can not be handled!!!", cmd_type);
    }
  }
} 

bool OpLogger::RestoreIterator(const std::string& name, void* ctx) {
  
  //if(saved_iters_.size()==iters_.size()) return true; 
  if(saved_iters_.find(name)!=saved_iters_.end()) return false; 

  // add it to recovered iterators
  saved_iters_[name] = NULL;

  VariantTensorData data;
  const char* path = getenv("TF_CHECKPOINT_ITER_SAVEPATH");
  std::string file_path = std::string(path)+name;
  fprintf(stderr, "Start to Restore Iterator %s\n", file_path.c_str());
  std::ifstream ifile;
  ifile.open(file_path.c_str(), std::ifstream::binary);
  if (!ifile.is_open()){
    fprintf(stderr, "Fail to open %s\n", file_path.c_str());
    return false;
  } 

  ifile.seekg(0, ifile.end);
  size_t size = ifile.tellg();
  ifile.seekg(0, ifile.beg);
  std::string saved_data(size, '\0');
  //saved_data.reserve(size);
  ifile.read(const_cast<char*>(saved_data.data()), size);
  //FIX: This approach only reads one line
  //std::string saved_data;
  //ifile >> saved_data;
  ifile.close();
  //fprintf(stderr, "saved_data %s\n", saved_data.c_str());
  if(!data.ParseFromString(saved_data)) {
    fprintf(stderr, "fail to parse saved_data\n");
    return false;
  }

  data::VariantTensorDataReader reader(&data);
  IteratorBase* iterbase = (IteratorBase*)iters_[name]; 
  fprintf(stderr, "Before Restore Iterator %s\n", name.c_str());
  Status s = iterbase->Restore((IteratorContext*)ctx, &reader); 
  if(!s.ok()) {
    fprintf(stderr, "WARNING: restore iterator failed: %s\n", name.c_str());
  }
  fprintf(stderr, "Restore Iterator %s\n", name.c_str());
  
  return false;
}

void OpLogger::SaveIterators() {
  mutex_lock lock(sit_mu_);
  bool current_state = in_library;
  if(!current_state) {
    in_library = true;
  }
  /*
  fprintf(stderr, "=============================\n");
  for(const auto& iter : iters_) {
    fprintf(stderr, "Iter %p, %s\n", iter.second, iter.first.c_str());
  }
  */
  const char* cmd = getenv("TF_CHECKPOINT_ITER_CMD");
  //if(cmd==NULL && strcmp(cmd, "2") == 0) { 
  if(strcmp(cmd, "2") == 0) { 
    const char* path = getenv("TF_CHECKPOINT_ITER_SAVEPATH");
    fprintf(stderr, "=============================\n");
    for(const auto& iter : iters_) {
      fprintf(stderr, "Save Iter %p, %s at %s\n", iter.second, iter.first.c_str(), path);
      VariantTensorData data;
      data.set_type_name(iter.first);
      data::VariantTensorDataWriter writer(&data);
      SerializationContext serialization_ctx({});
      IteratorBase* iterbase = (IteratorBase*)iter.second; 
      Status s = iterbase->Save(&serialization_ctx, &writer);
      if(!s.ok()) {
        fprintf(stderr, "WARNING: Save iterator failed!!! %s\n", iter.first.c_str());
      }
      writer.Flush();

      std::string data_str = data.SerializeAsString();
      size_t size = data_str.size();

      std::string file_path = std::string(path)+iter.first;
      std::ofstream ofile;
      ofile.open(file_path.c_str(), std::ofstream::trunc | std::ofstream::binary);
      ofile.write(const_cast<char*>(data_str.data()), size);
      //ofile << data.SerializeAsString().c_str() << "\0";
      ofile.close();
    }
    for(const auto& iter : del_iters_) {
      std::string file_path = std::string(path)+iter.first;
      std::ifstream srce(iter.second.c_str(), std::ios::binary);
      std::ofstream dest(file_path.c_str(), std::ios::binary);
      dest << srce.rdbuf();
      srce.close();
      dest.close();
    }
    
    setenv("TF_CHECKPOINT_ITER_CMD", "0", 1);
  }
  if(!current_state) {
    in_library = false;
  }
}

