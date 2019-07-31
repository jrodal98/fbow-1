// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: agent_brain.proto

#ifndef PROTOBUF_INCLUDED_agent_5fbrain_2eproto
#define PROTOBUF_INCLUDED_agent_5fbrain_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_agent_5fbrain_2eproto 

namespace protobuf_agent_5fbrain_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[3];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_agent_5fbrain_2eproto
namespace agent_brain {
class slam_data;
class slam_dataDefaultTypeInternal;
extern slam_dataDefaultTypeInternal _slam_data_default_instance_;
class slam_data_bin_description;
class slam_data_bin_descriptionDefaultTypeInternal;
extern slam_data_bin_descriptionDefaultTypeInternal _slam_data_bin_description_default_instance_;
class slam_data_keypoint;
class slam_data_keypointDefaultTypeInternal;
extern slam_data_keypointDefaultTypeInternal _slam_data_keypoint_default_instance_;
}  // namespace agent_brain
namespace google {
namespace protobuf {
template<> ::agent_brain::slam_data* Arena::CreateMaybeMessage<::agent_brain::slam_data>(Arena*);
template<> ::agent_brain::slam_data_bin_description* Arena::CreateMaybeMessage<::agent_brain::slam_data_bin_description>(Arena*);
template<> ::agent_brain::slam_data_keypoint* Arena::CreateMaybeMessage<::agent_brain::slam_data_keypoint>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace agent_brain {

// ===================================================================

class slam_data_keypoint : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:agent_brain.slam_data.keypoint) */ {
 public:
  slam_data_keypoint();
  virtual ~slam_data_keypoint();

  slam_data_keypoint(const slam_data_keypoint& from);

  inline slam_data_keypoint& operator=(const slam_data_keypoint& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  slam_data_keypoint(slam_data_keypoint&& from) noexcept
    : slam_data_keypoint() {
    *this = ::std::move(from);
  }

  inline slam_data_keypoint& operator=(slam_data_keypoint&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const slam_data_keypoint& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const slam_data_keypoint* internal_default_instance() {
    return reinterpret_cast<const slam_data_keypoint*>(
               &_slam_data_keypoint_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(slam_data_keypoint* other);
  friend void swap(slam_data_keypoint& a, slam_data_keypoint& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline slam_data_keypoint* New() const final {
    return CreateMaybeMessage<slam_data_keypoint>(NULL);
  }

  slam_data_keypoint* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<slam_data_keypoint>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const slam_data_keypoint& from);
  void MergeFrom(const slam_data_keypoint& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(slam_data_keypoint* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated uint32 bgr = 3;
  int bgr_size() const;
  void clear_bgr();
  static const int kBgrFieldNumber = 3;
  ::google::protobuf::uint32 bgr(int index) const;
  void set_bgr(int index, ::google::protobuf::uint32 value);
  void add_bgr(::google::protobuf::uint32 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >&
      bgr() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >*
      mutable_bgr();

  // uint32 x = 1;
  void clear_x();
  static const int kXFieldNumber = 1;
  ::google::protobuf::uint32 x() const;
  void set_x(::google::protobuf::uint32 value);

  // uint32 y = 2;
  void clear_y();
  static const int kYFieldNumber = 2;
  ::google::protobuf::uint32 y() const;
  void set_y(::google::protobuf::uint32 value);

  // @@protoc_insertion_point(class_scope:agent_brain.slam_data.keypoint)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< ::google::protobuf::uint32 > bgr_;
  mutable int _bgr_cached_byte_size_;
  ::google::protobuf::uint32 x_;
  ::google::protobuf::uint32 y_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_agent_5fbrain_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class slam_data_bin_description : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:agent_brain.slam_data.bin_description) */ {
 public:
  slam_data_bin_description();
  virtual ~slam_data_bin_description();

  slam_data_bin_description(const slam_data_bin_description& from);

  inline slam_data_bin_description& operator=(const slam_data_bin_description& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  slam_data_bin_description(slam_data_bin_description&& from) noexcept
    : slam_data_bin_description() {
    *this = ::std::move(from);
  }

  inline slam_data_bin_description& operator=(slam_data_bin_description&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const slam_data_bin_description& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const slam_data_bin_description* internal_default_instance() {
    return reinterpret_cast<const slam_data_bin_description*>(
               &_slam_data_bin_description_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(slam_data_bin_description* other);
  friend void swap(slam_data_bin_description& a, slam_data_bin_description& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline slam_data_bin_description* New() const final {
    return CreateMaybeMessage<slam_data_bin_description>(NULL);
  }

  slam_data_bin_description* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<slam_data_bin_description>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const slam_data_bin_description& from);
  void MergeFrom(const slam_data_bin_description& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(slam_data_bin_description* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string description = 1;
  void clear_description();
  static const int kDescriptionFieldNumber = 1;
  const ::std::string& description() const;
  void set_description(const ::std::string& value);
  #if LANG_CXX11
  void set_description(::std::string&& value);
  #endif
  void set_description(const char* value);
  void set_description(const char* value, size_t size);
  ::std::string* mutable_description();
  ::std::string* release_description();
  void set_allocated_description(::std::string* description);

  // @@protoc_insertion_point(class_scope:agent_brain.slam_data.bin_description)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr description_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_agent_5fbrain_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class slam_data : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:agent_brain.slam_data) */ {
 public:
  slam_data();
  virtual ~slam_data();

  slam_data(const slam_data& from);

  inline slam_data& operator=(const slam_data& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  slam_data(slam_data&& from) noexcept
    : slam_data() {
    *this = ::std::move(from);
  }

  inline slam_data& operator=(slam_data&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const slam_data& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const slam_data* internal_default_instance() {
    return reinterpret_cast<const slam_data*>(
               &_slam_data_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void Swap(slam_data* other);
  friend void swap(slam_data& a, slam_data& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline slam_data* New() const final {
    return CreateMaybeMessage<slam_data>(NULL);
  }

  slam_data* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<slam_data>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const slam_data& from);
  void MergeFrom(const slam_data& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(slam_data* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef slam_data_keypoint keypoint;
  typedef slam_data_bin_description bin_description;

  // accessors -------------------------------------------------------

  // repeated .agent_brain.slam_data.keypoint keypoints = 1;
  int keypoints_size() const;
  void clear_keypoints();
  static const int kKeypointsFieldNumber = 1;
  ::agent_brain::slam_data_keypoint* mutable_keypoints(int index);
  ::google::protobuf::RepeatedPtrField< ::agent_brain::slam_data_keypoint >*
      mutable_keypoints();
  const ::agent_brain::slam_data_keypoint& keypoints(int index) const;
  ::agent_brain::slam_data_keypoint* add_keypoints();
  const ::google::protobuf::RepeatedPtrField< ::agent_brain::slam_data_keypoint >&
      keypoints() const;

  // repeated string descriptions = 2;
  int descriptions_size() const;
  void clear_descriptions();
  static const int kDescriptionsFieldNumber = 2;
  const ::std::string& descriptions(int index) const;
  ::std::string* mutable_descriptions(int index);
  void set_descriptions(int index, const ::std::string& value);
  #if LANG_CXX11
  void set_descriptions(int index, ::std::string&& value);
  #endif
  void set_descriptions(int index, const char* value);
  void set_descriptions(int index, const char* value, size_t size);
  ::std::string* add_descriptions();
  void add_descriptions(const ::std::string& value);
  #if LANG_CXX11
  void add_descriptions(::std::string&& value);
  #endif
  void add_descriptions(const char* value);
  void add_descriptions(const char* value, size_t size);
  const ::google::protobuf::RepeatedPtrField< ::std::string>& descriptions() const;
  ::google::protobuf::RepeatedPtrField< ::std::string>* mutable_descriptions();

  // @@protoc_insertion_point(class_scope:agent_brain.slam_data)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::agent_brain::slam_data_keypoint > keypoints_;
  ::google::protobuf::RepeatedPtrField< ::std::string> descriptions_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_agent_5fbrain_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// slam_data_keypoint

// uint32 x = 1;
inline void slam_data_keypoint::clear_x() {
  x_ = 0u;
}
inline ::google::protobuf::uint32 slam_data_keypoint::x() const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.keypoint.x)
  return x_;
}
inline void slam_data_keypoint::set_x(::google::protobuf::uint32 value) {
  
  x_ = value;
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.keypoint.x)
}

// uint32 y = 2;
inline void slam_data_keypoint::clear_y() {
  y_ = 0u;
}
inline ::google::protobuf::uint32 slam_data_keypoint::y() const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.keypoint.y)
  return y_;
}
inline void slam_data_keypoint::set_y(::google::protobuf::uint32 value) {
  
  y_ = value;
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.keypoint.y)
}

// repeated uint32 bgr = 3;
inline int slam_data_keypoint::bgr_size() const {
  return bgr_.size();
}
inline void slam_data_keypoint::clear_bgr() {
  bgr_.Clear();
}
inline ::google::protobuf::uint32 slam_data_keypoint::bgr(int index) const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.keypoint.bgr)
  return bgr_.Get(index);
}
inline void slam_data_keypoint::set_bgr(int index, ::google::protobuf::uint32 value) {
  bgr_.Set(index, value);
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.keypoint.bgr)
}
inline void slam_data_keypoint::add_bgr(::google::protobuf::uint32 value) {
  bgr_.Add(value);
  // @@protoc_insertion_point(field_add:agent_brain.slam_data.keypoint.bgr)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >&
slam_data_keypoint::bgr() const {
  // @@protoc_insertion_point(field_list:agent_brain.slam_data.keypoint.bgr)
  return bgr_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >*
slam_data_keypoint::mutable_bgr() {
  // @@protoc_insertion_point(field_mutable_list:agent_brain.slam_data.keypoint.bgr)
  return &bgr_;
}

// -------------------------------------------------------------------

// slam_data_bin_description

// string description = 1;
inline void slam_data_bin_description::clear_description() {
  description_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& slam_data_bin_description::description() const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.bin_description.description)
  return description_.GetNoArena();
}
inline void slam_data_bin_description::set_description(const ::std::string& value) {
  
  description_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.bin_description.description)
}
#if LANG_CXX11
inline void slam_data_bin_description::set_description(::std::string&& value) {
  
  description_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:agent_brain.slam_data.bin_description.description)
}
#endif
inline void slam_data_bin_description::set_description(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  description_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:agent_brain.slam_data.bin_description.description)
}
inline void slam_data_bin_description::set_description(const char* value, size_t size) {
  
  description_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:agent_brain.slam_data.bin_description.description)
}
inline ::std::string* slam_data_bin_description::mutable_description() {
  
  // @@protoc_insertion_point(field_mutable:agent_brain.slam_data.bin_description.description)
  return description_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* slam_data_bin_description::release_description() {
  // @@protoc_insertion_point(field_release:agent_brain.slam_data.bin_description.description)
  
  return description_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void slam_data_bin_description::set_allocated_description(::std::string* description) {
  if (description != NULL) {
    
  } else {
    
  }
  description_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), description);
  // @@protoc_insertion_point(field_set_allocated:agent_brain.slam_data.bin_description.description)
}

// -------------------------------------------------------------------

// slam_data

// repeated .agent_brain.slam_data.keypoint keypoints = 1;
inline int slam_data::keypoints_size() const {
  return keypoints_.size();
}
inline void slam_data::clear_keypoints() {
  keypoints_.Clear();
}
inline ::agent_brain::slam_data_keypoint* slam_data::mutable_keypoints(int index) {
  // @@protoc_insertion_point(field_mutable:agent_brain.slam_data.keypoints)
  return keypoints_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::agent_brain::slam_data_keypoint >*
slam_data::mutable_keypoints() {
  // @@protoc_insertion_point(field_mutable_list:agent_brain.slam_data.keypoints)
  return &keypoints_;
}
inline const ::agent_brain::slam_data_keypoint& slam_data::keypoints(int index) const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.keypoints)
  return keypoints_.Get(index);
}
inline ::agent_brain::slam_data_keypoint* slam_data::add_keypoints() {
  // @@protoc_insertion_point(field_add:agent_brain.slam_data.keypoints)
  return keypoints_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::agent_brain::slam_data_keypoint >&
slam_data::keypoints() const {
  // @@protoc_insertion_point(field_list:agent_brain.slam_data.keypoints)
  return keypoints_;
}

// repeated string descriptions = 2;
inline int slam_data::descriptions_size() const {
  return descriptions_.size();
}
inline void slam_data::clear_descriptions() {
  descriptions_.Clear();
}
inline const ::std::string& slam_data::descriptions(int index) const {
  // @@protoc_insertion_point(field_get:agent_brain.slam_data.descriptions)
  return descriptions_.Get(index);
}
inline ::std::string* slam_data::mutable_descriptions(int index) {
  // @@protoc_insertion_point(field_mutable:agent_brain.slam_data.descriptions)
  return descriptions_.Mutable(index);
}
inline void slam_data::set_descriptions(int index, const ::std::string& value) {
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.descriptions)
  descriptions_.Mutable(index)->assign(value);
}
#if LANG_CXX11
inline void slam_data::set_descriptions(int index, ::std::string&& value) {
  // @@protoc_insertion_point(field_set:agent_brain.slam_data.descriptions)
  descriptions_.Mutable(index)->assign(std::move(value));
}
#endif
inline void slam_data::set_descriptions(int index, const char* value) {
  GOOGLE_DCHECK(value != NULL);
  descriptions_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:agent_brain.slam_data.descriptions)
}
inline void slam_data::set_descriptions(int index, const char* value, size_t size) {
  descriptions_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:agent_brain.slam_data.descriptions)
}
inline ::std::string* slam_data::add_descriptions() {
  // @@protoc_insertion_point(field_add_mutable:agent_brain.slam_data.descriptions)
  return descriptions_.Add();
}
inline void slam_data::add_descriptions(const ::std::string& value) {
  descriptions_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:agent_brain.slam_data.descriptions)
}
#if LANG_CXX11
inline void slam_data::add_descriptions(::std::string&& value) {
  descriptions_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:agent_brain.slam_data.descriptions)
}
#endif
inline void slam_data::add_descriptions(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  descriptions_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:agent_brain.slam_data.descriptions)
}
inline void slam_data::add_descriptions(const char* value, size_t size) {
  descriptions_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:agent_brain.slam_data.descriptions)
}
inline const ::google::protobuf::RepeatedPtrField< ::std::string>&
slam_data::descriptions() const {
  // @@protoc_insertion_point(field_list:agent_brain.slam_data.descriptions)
  return descriptions_;
}
inline ::google::protobuf::RepeatedPtrField< ::std::string>*
slam_data::mutable_descriptions() {
  // @@protoc_insertion_point(field_mutable_list:agent_brain.slam_data.descriptions)
  return &descriptions_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace agent_brain

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_agent_5fbrain_2eproto
