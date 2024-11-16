package org.hit.jdbcdemo;

import org.hit.jdbcdemo.mapper.OrderMapper;
import org.hit.jdbcdemo.mapper.UserMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.hit.jdbcdemo.entity.Order;

import java.math.BigDecimal;

@SpringBootTest
public class ShardingTest {

    @Autowired
    private UserMapper userMapper;

    @Autowired
    private OrderMapper orderMapper;

    /**
     * 水平分片：分库插入数据测试
     */
    @Test
    public void testInsertOrderDatabaseStrategy(){

        for (long i = 0; i < 4; i++) {

            Order order = new Order();
            order.setOrderNo("软工学员"+i+"号");
            order.setUserId(i + 1);
            order.setAmount(new BigDecimal(100));
            orderMapper.insert(order);
        }
    }

    /**
     * 水平分片：分表插入数据测试
     */
    @Test
    public void testInsertOrderTableStrategy(){

        for (long i = 1; i < 5; i++) {

            Order order = new Order();
            order.setOrderNo("计算学部学生" + i);
            order.setUserId(1L);
            order.setAmount(new BigDecimal(100));
            orderMapper.insert(order);
        }

        for (long i = 5; i < 9; i++) {

            Order order = new Order();
            order.setOrderNo("计算学部学生" + i);
            order.setUserId(2L);
            order.setAmount(new BigDecimal(100));
            orderMapper.insert(order);
        }
    }
}
