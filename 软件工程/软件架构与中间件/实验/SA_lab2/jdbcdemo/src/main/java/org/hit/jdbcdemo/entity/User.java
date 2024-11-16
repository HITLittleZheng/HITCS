package org.hit.jdbcdemo.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@TableName("t_user")//逻辑表
@Data
public class User {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String uname;
}
