package org.hit.jdbcdemo.mapper;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.hit.jdbcdemo.entity.User;

@Mapper
public interface UserMapper extends BaseMapper<User> {
}
