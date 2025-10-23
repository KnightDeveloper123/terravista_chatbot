import React from 'react'
import { useForm, SubmitHandler } from 'react-hook-form'
import { useNavigate } from 'react-router'
import useFetch from '@/hooks/useFetch'
import {
    Box,
    Button,
    Field,
    Fieldset,
    Flex,
    Heading,
    Input,
    Stack,
    Text
} from '@chakra-ui/react'
import { showAlert } from '@/utils/helpers'

interface RegisterFormInputs {
    name: string
    email: string
    password: string
    confirmPassword: string
}

const Register: React.FC = () => {
    const {
        register,
        handleSubmit,
        formState: { errors },
        watch,
    } = useForm<RegisterFormInputs>()

    const { request, loading } = useFetch()
    const navigate = useNavigate()

    const onSubmit: SubmitHandler<RegisterFormInputs> = async (data) => {
        if (data.password !== data.confirmPassword) {
            showAlert("Error", "Passwords do not match", "error")
            return
        }

        try {
            const response = await request({
                url: "/user/signUp",
                method: "POST",
                body: {
                    name: data.name,
                    email: data.email,
                    password: data.password,
                },
            })

            if (!response || response.error) {
                showAlert("Registration Failed", response?.error || "Something went wrong", "error")
                return
            }

            showAlert("Success", "Account created successfully! Please login.", "success")
            navigate("/auth/login")
        } catch (err: any) {
            showAlert("Registration Failed", err?.message || "Something went wrong", "error")
        }
    }

    return (
        <Box bg="#f5f9ff" minH="100vh">
            <Flex justify="center" align="center" h="100vh" position="relative">
                <Flex
                    maxW="400px"
                    w="full"
                    bg="#fff"
                    color="black"
                    justify="center"
                    align="center"
                    direction="column"
                    borderRadius="lg"
                    border="1px solid #00539230"
                    boxShadow="0 0 12px rgba(0,83,146,0.15)"
                    py={8}
                    px={6}
                >
                    <Heading mb={6} fontSize="2xl" color="#005392">
                        Register
                    </Heading>

                    <form onSubmit={handleSubmit(onSubmit)} style={{ width: "100%" }}>
                        <Fieldset.Root w="100%">
                            <Stack gap={5}>
                                {/* Name */}
                                <Field.Root>
                                    <Field.Label>Name</Field.Label>
                                    <Input
                                        placeholder="Your name"
                                        border="1px solid #005392"
                                        borderRadius="8px"
                                        _focus={{
                                            outline: 'none',
                                            borderColor: '#005392',
                                            boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.6)',
                                        }}
                                        {...register("name", { required: "Name is required" })}
                                    />
                                    {errors.name && (
                                        <Text color="red.500" fontSize="sm" mt={1}>
                                            {errors.name.message}
                                        </Text>
                                    )}
                                </Field.Root>

                                {/* Email */}
                                <Field.Root>
                                    <Field.Label>Email</Field.Label>
                                    <Input
                                        placeholder="eg: john@example.com"
                                        type="email"
                                        border="1px solid #005392"
                                        borderRadius="8px"
                                        _focus={{
                                            outline: 'none',
                                            borderColor: '#005392',
                                            boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.6)',
                                        }}
                                        {...register("email", {
                                            required: "Email is required",
                                            pattern: {
                                                value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
                                                message: "Invalid email format",
                                            },
                                        })}
                                    />
                                    {errors.email && (
                                        <Text color="red.500" fontSize="sm" mt={1}>
                                            {errors.email.message}
                                        </Text>
                                    )}
                                </Field.Root>

                                {/* Password */}
                                <Field.Root>
                                    <Field.Label>Password</Field.Label>
                                    <Input
                                        placeholder="Enter your password"
                                        type="password"
                                        border="1px solid #005392"
                                        borderRadius="8px"
                                        _focus={{
                                            outline: 'none',
                                            borderColor: '#005392',
                                            boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.6)',
                                        }}
                                        {...register("password", {
                                            required: "Password is required",
                                            minLength: {
                                                value: 6,
                                                message: "Password must be at least 6 characters",
                                            },
                                        })}
                                    />
                                    {errors.password && (
                                        <Text color="red.500" fontSize="sm" mt={1}>
                                            {errors.password.message}
                                        </Text>
                                    )}
                                </Field.Root>

                                {/* Confirm Password */}
                                <Field.Root>
                                    <Field.Label>Confirm Password</Field.Label>
                                    <Input
                                        placeholder="Confirm password"
                                        type="password"
                                        border="1px solid #005392"
                                        borderRadius="8px"
                                        _focus={{
                                            outline: 'none',
                                            borderColor: '#005392',
                                            boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.6)',
                                        }}
                                        {...register("confirmPassword", {
                                            required: "Please confirm your password",
                                            validate: (val) => val === watch("password") || "Passwords do not match",
                                        })}
                                    />
                                    {errors.confirmPassword && (
                                        <Text color="red.500" fontSize="sm" mt={1}>
                                            {errors.confirmPassword.message}
                                        </Text>
                                    )}
                                </Field.Root>

                                <Button
                                    type="submit"
                                    mt={2}
                                    width="100%"
                                    bg="#005392"
                                    color="white"
                                    borderRadius="full"
                                    loading={loading}
                                    loadingText="Registering..."
                                    _hover={{ bg: '#00406b' }}
                                    _active={{ bg: '#003357' }}
                                >
                                    Register
                                </Button>
                            </Stack>
                        </Fieldset.Root>
                    </form>
                    <Text fontSize={'12px'} mt={2} _hover={{ color: "#005392", cursor: 'pointer' }} onClick={() => navigate('/auth/login')}>Already have an account?</Text>

                </Flex>
            </Flex>
        </Box>
    )
}

export default Register
