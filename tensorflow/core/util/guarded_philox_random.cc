/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include <fstream>

namespace tensorflow {

Status GuardedPhiloxRandom::Init(OpKernelConstruction* context) {
  // Grab seed Attrs.
  int64 seed, seed2;
  auto status = context->GetAttr("seed", &seed);
  if (!status.ok()) return status;
  status = context->GetAttr("seed2", &seed2);
  if (!status.ok()) return status;

  // Initialize with the given seeds
  Init(seed, seed2);

  std::size_t found = context->def().name().find("random_uniform/RandomUniform");
  std::size_t found1 = context->def().name().find("dropout");
  if(found!=std::string::npos && found1!=std::string::npos) {
    RandomMonitor::getInstance().AddRandom(&generator_);
    //fprintf(stderr, "GuardedPhiloxRandom::Init: op [%s], seed %zu, seed2 %zu\n", context->def().name().c_str(), seed, seed2);
  }
  return Status::OK();
}

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  CHECK(!initialized_);
  this->seed = seed;
  this->seed2 = seed2;
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  } 
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(seed, seed2);
  initialized_ = true;
}

void GuardedPhiloxRandom::Init(random::PhiloxRandom::ResultType counter,
                               random::PhiloxRandom::Key key) {
  CHECK(!initialized_);
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(counter, key);
  initialized_ = true;
}

random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  CHECK(initialized_);
  mutex_lock lock(mu_);

  RandomMonitor::getInstance().Monitor(&generator_);

  auto local = generator_;
  generator_.Skip(samples);
  return local;
}

random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples, int64 local_seed) {
  mutex_lock lock(mu1_);
  if(seed==0) {
    // stateless random operation may not need to be recorded
    CHECK(initialized_);
    mutex_lock lock(mu_);
    auto local = generator_;
    generator_.Skip(samples);
    return local;
  } else {
    //fprintf(stderr, "\nseed %ld, seed2 %ld, local seed %lx\n", seed, seed2, local_seed);
    generator1_ = random::PhiloxRandom(seed2, local_seed);
    auto local = generator1_;
    generator1_.Skip(samples);
    return local;
  }
}

void RandomMonitor::Monitor(random::PhiloxRandom* rng) {
  const char* cmd = getenv("TF_CHECKPOINT_RNG_CMD");
  if(cmd!=NULL){
    if(strcmp(cmd, "1") == 0) {
      LoadState();
      RestoreRandom(rng);
      setenv("TF_CHECKPOINT_RNG_CMD", "3", 1);
    } 
    else if(strcmp(cmd, "2") == 0) {
      SaveState();
      setenv("TF_CHECKPOINT_RNG_CMD", "0", 1);
    }
    else if(strcmp(cmd, "3") == 0) {
      RestoreRandom(rng);
    }
  }
}

void RandomMonitor::LoadState() {
  mutex_lock lock(mu_);
  const char* cmd = getenv("TF_CHECKPOINT_RNG_CMD");
  if(strcmp(cmd, "1") == 0) {
    const char* path = getenv("TF_CHECKPOINT_RNG_STATE");
    std::ifstream ifile;
    ifile.open(path);
    if (!ifile.is_open())
      return;

    std::string id;
    while(ifile >> id){
      history[id] = std::vector<uint32>(6);
      for(int i=0; i<6; i++) {
        uint32 val;
        ifile >> val;
        history[id][i] = val;
      }
    }
    ifile.close();
  }
}

void RandomMonitor::SaveState() {
  mutex_lock lock(mu_);
  std::ofstream ofile;
  const char* path = getenv("TF_CHECKPOINT_RNG_STATE");
  ofile.open(path, std::ofstream::out | std::ofstream::trunc);
  for(const auto& rng : rngs) {
    ofile << rng.second << ' ';
    random::PhiloxRandom* rand = (random::PhiloxRandom*)rng.first;
    random::PhiloxRandom::Key const& key = rand->key();
    random::PhiloxRandom::ResultType const& counter = rand->counter();
    ofile << key[0] << ' ' << key[1] << ' ';
    ofile << counter[0] << ' ' << counter[1] << ' ' << counter[2] << ' ' << counter[3];
    ofile << std::endl;
  }
  ofile.close();
}

void RandomMonitor::AddRandom(random::PhiloxRandom* rng) {
  mutex_lock lock(mu_);
  random::PhiloxRandom::Key const& key = rng->key();
  random::PhiloxRandom::ResultType const& counter = rng->counter();
  rngs[rng] = std::string(
      std::to_string(key[0])+"_"+
      std::to_string(key[1])+"_"+
      std::to_string(counter[2])+"_"+
      std::to_string(counter[3]));
}

void RandomMonitor::RestoreRandom(random::PhiloxRandom* rng) {
  mutex_lock lock(mu_);
  std::vector<uint32>& val = history[rngs[rng]];
  if(val.size() != 6) return;

  random::PhiloxRandom::Key key;
  random::PhiloxRandom::ResultType counter;
  key[0] = val[0];
  key[1] = val[1];
  counter[0] = val[2];
  counter[1] = val[3];
  counter[2] = val[4];
  counter[3] = val[5];
  rng->ResetRandom(key, counter);
 
  history.erase(rngs[rng]);
}

}  // namespace tensorflow
